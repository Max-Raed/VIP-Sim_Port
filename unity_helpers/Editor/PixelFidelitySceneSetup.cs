// PixelFidelitySceneSetup.cs — one-shot Unity Editor script that auto-builds
// the PixelFidelity capture scene for the AVP / VipSim pixel-fidelity pipeline.
//
// Run from Unity Editor: menu  VipSim → Setup PixelFidelity Scene
//
// What it does (idempotent — re-running cleans up first):
//   1. Ensures the "LeftEye" and "RightEye" tags exist in TagManager.
//   2. Adds every shader under Assets/VisualEffects/Shaders/ to
//      Project Settings → Graphics → Always Included Shaders.
//      (Fixes Hidden/VisSim/* shader stripping for both primary + secondary
//       shaders loaded via Shader.Find at runtime, e.g. cfxFrost.)
//   3. Deletes any existing LeftEye / RightEye / Capture / InputQuad in the
//      active scene, then creates fresh ones:
//        - LeftEye: tagged "LeftEye", orthographic Camera at (0,0,-10), size 5
//        - RightEye: tagged "RightEye", disabled Camera (required by
//          LinkableBaseEffect, which checks both eyes)
//        - InputQuad: 20×10 unlit quad bound to Trafficscene_2048x1024.png
//          if found at vipsim_assets/Trafficscene_2048x1024.png or
//          Assets/VisualEffects/Trafficscene_2048x1024.png
//        - Capture: GameObject with PixelFidelityCapture, captureCamera wired
//   4. AddComponent of every concrete subclass of VisSim.LinkableBaseEffect on
//      both LeftEye and RightEye, all disabled. Inpainter* and
//      myFieldLossInverted are skipped.
//
// Setup (one-time per Unity project):
//   - Drop this file under any folder named "Editor" in Assets/
//     (e.g. Assets/AVP/Editor/PixelFidelitySceneSetup.cs).
//   - Drop PixelFidelityCapture.cs anywhere in Assets/ (NOT under Editor/).
//   - Open Unity. Menu → VipSim → Setup PixelFidelity Scene.
//   - Save the scene as PixelFidelity.unity.

#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace AVP.EditorTools
{
    public static class PixelFidelitySceneSetup
    {
        // Where to look for shaders + input texture. Edit if your project uses a
        // different path.
        const string ShadersFolder = "Assets/VisualEffects/Shaders";
        static readonly string[] InputTextureCandidates =
        {
            "Assets/Resources/Inputs/Trafficscene_2048x1024.png",
            "Assets/VisualEffects/Trafficscene_2048x1024.png",
            "Assets/Trafficscene_2048x1024.png",
            "Assets/vipsim_assets/Trafficscene_2048x1024.png",
        };

        // Filters that extend MonoBehaviour directly (not LinkableBaseEffect),
        // so DiscoverEffectTypes wouldn't catch them. Image effects with their
        // own OnRenderImage. Attach by short name; namespace may vary.
        static readonly string[] ExtraEffectTypeNames =
        {
            "FlickeringStars",
            "FovealDarkness",
            "PixelationEffect",
            "VortexEffect",
        };

        // Filter classes to skip (depend on missing resources / out of scope).
        static readonly HashSet<string> SkipTypes = new HashSet<string>
        {
            "myInpainter", "myInpainter2", "myInpainter2_hacked",
            "myFieldLossInverted", // covered by myFieldLoss; remove if you want both
        };

        [MenuItem("VipSim/Setup PixelFidelity Scene")]
        public static void Setup()
        {
            EnsureTag("LeftEye");
            EnsureTag("RightEye");

            int shaderCount = RegisterAllShaders();

            // Build name -> Shader dictionary so we can populate the public
            // `Shader` field on every filter BEFORE OnEnable fires. Otherwise
            // BaseEffect.ShaderSafe falls back to Shader.Find(GetShaderName())
            // which can return null in the editor (race with reimport, etc.)
            // and causes `new Material(null)` to throw.
            var shaderByName = BuildShaderLookup();

            // Wipe any prior setup so re-running is idempotent. Delete BY TAG,
            // not just by name, otherwise an old Main Camera tagged LeftEye
            // survives and FindWithTag in LinkableBaseEffect.OnEnable returns
            // the wrong GameObject.
            DeleteAllWithTag("LeftEye");
            DeleteAllWithTag("RightEye");
            DeleteByName("InputQuad");
            DeleteByName("Capture");
            DeleteByName("GazeTracker");

            // GazeTracker singleton: needed by FovealDarkness, VortexEffect,
            // myDistortionMap, myTeichopsia, myFieldLoss, myFloaters, etc.
            // GazeSource.None pins xy_norm at (0.5, 0.5) — fixed center gaze
            // for reproducible validation.
            var gazeTrackerType = FindType("GazeTracker");
            if (gazeTrackerType != null)
            {
                var gtGO = new GameObject("GazeTracker");
                var gt = gtGO.AddComponent(gazeTrackerType) as MonoBehaviour;
                var gazeSourceField = gazeTrackerType.GetField("gazeSource");
                if (gazeSourceField != null && gazeSourceField.FieldType.IsEnum)
                {
                    // None = 3 (per the GazeSource enum: Fove=0, UnitEye=1, Mouse=2, None=3)
                    gazeSourceField.SetValue(gt, System.Enum.ToObject(gazeSourceField.FieldType, 3));
                }
            }
            else
            {
                Debug.LogWarning("[PixelFidelitySetup] GazeTracker type not found. Several gaze-contingent filters will throw NullRefs in OnRenderImage / OnUpdate.");
            }

            // LeftEye = capture camera. Created INACTIVE so AddComponent does
            // not fire OnEnable on the filter components before their Shader
            // field is populated.
            var leftEye = new GameObject("LeftEye");
            leftEye.SetActive(false);
            leftEye.tag = "LeftEye";
            var leftCam = leftEye.AddComponent<Camera>();
            leftCam.orthographic = true;
            leftCam.orthographicSize = 5f;
            leftCam.clearFlags = CameraClearFlags.SolidColor;
            leftCam.backgroundColor = Color.black;
            leftEye.transform.position = new Vector3(0, 0, -10);

            // RightEye = required sibling for LinkableBaseEffect (kept disabled).
            var rightEye = new GameObject("RightEye");
            rightEye.SetActive(false);
            rightEye.tag = "RightEye";
            var rightCam = rightEye.AddComponent<Camera>();
            rightCam.enabled = false;

            // Attach every concrete VipSim filter to both eyes, disabled.
            // GameObjects are inactive here, so OnEnable doesn't fire yet.
            var effectTypes = DiscoverEffectTypes();
            foreach (var name in ExtraEffectTypeNames)
            {
                var t = FindType(name);
                if (t != null && !effectTypes.Contains(t)) effectTypes.Add(t);
            }

            int attached = 0;
            foreach (var t in effectTypes.OrderBy(t => t.Name))
            {
                if (SkipTypes.Contains(t.Name)) continue;
                AttachDisabledWithShader(leftEye, t, shaderByName);
                AttachDisabledWithShader(rightEye, t, shaderByName);
                attached++;
            }

            // Now that both eyes have all components AND each has its Shader
            // field populated, activate. OnEnable on every component will run
            // here; LinkableBaseEffect's pair check passes because both eyes
            // already have the component, and Material construction succeeds
            // because Shader is non-null.
            leftEye.SetActive(true);
            rightEye.SetActive(true);

            // Optional InputQuad bound to the canonical Trafficscene texture.
            CreateInputQuadIfTexturePresent();

            // Capture GameObject with PixelFidelityCapture wired to LeftEye.
            int specsAdded = 0;
            var captureType = FindType("PixelFidelityCapture");
            if (captureType != null)
            {
                var captureGO = new GameObject("Capture");
                var capture = captureGO.AddComponent(captureType) as MonoBehaviour;
                var camField = captureType.GetField("captureCamera",
                    System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
                camField?.SetValue(capture, leftCam);

                specsAdded = PopulateDefaultSpecs(capture, captureType, effectTypes);
            }
            else
            {
                Debug.LogWarning("[PixelFidelitySetup] PixelFidelityCapture type not found. " +
                                 "Drop PixelFidelityCapture.cs into Assets/ (NOT under an Editor/ folder) and re-run.");
            }

            var scene = EditorSceneManager.GetActiveScene();
            EditorSceneManager.MarkSceneDirty(scene);

            Debug.Log($"[PixelFidelitySetup] Done. Filters attached per eye: {attached}. " +
                      $"Shaders registered as Always Included: {shaderCount}. " +
                      $"Capture specs populated: {specsAdded}. " +
                      $"Save the scene (Ctrl+S) as PixelFidelity.unity.");
        }

        // Per-filter default Unity parameters for the first validation pass.
        // Match your Python severity ~0.4 / blur cpd 28. Edit values as needed
        // on the Capture GameObject in the Inspector after running.
        struct SpecTemplate
        {
            public string componentName;
            public string outputName;
            public (string name, float value)[] floatFields;
            public float extraWaitSeconds;
        }

        static readonly SpecTemplate[] DefaultSpecs = new[]
        {
            new SpecTemplate { componentName = "myBlur",                    outputName = "blur_cpd28.png",          floatFields = new[] { ("maxCPD", 28f) } },
            new SpecTemplate { componentName = "myBrightnessContrastGamma", outputName = "bcg_b-25_c25.png",        floatFields = new[] { ("Brightness", -25f), ("Contrast", 25f) } },
            new SpecTemplate { componentName = "myRecolour",                outputName = "cvd_deutan_sev06.png",    floatFields = new[] { ("anomType", 1f), ("severityIndex", 0.6f) } },
            new SpecTemplate { componentName = "myCataract",                outputName = "cataract_sev04.png",      floatFields = new[] { ("severityIndex", 0.4f) } },
            new SpecTemplate { componentName = "myBloom",                   outputName = "bloom_int20_thr03.png",   floatFields = new[] { ("intensity", 2.0f), ("threshold", 0.3f) } },
            new SpecTemplate { componentName = "myFieldLoss",               outputName = "fieldloss_default.png",   floatFields = new (string, float)[0] },
            new SpecTemplate { componentName = "myDistortionMap",           outputName = "distortion_default.png",  floatFields = new (string, float)[0] },
            // myDoubleVision: keep the binocular default capture for completeness.
            // The monocular path (IsMonocular=true) is broken in upstream VIP-Sim
            // — it references Hidden/VisSim/BasicShader which is a 3D surface
            // shader, not an image effect, so Graphics.Blit produces black.
            // For monocular double-vision validation we use DoubleVisionEffect
            // (separate component with the working Hidden/DoubleVision shader)
            // captured to doublevisioneffect.png.
            new SpecTemplate { componentName = "myDoubleVision",            outputName = "doublevision_default.png",floatFields = new (string, float)[0] },
            new SpecTemplate { componentName = "DoubleVisionEffect",        outputName = "doublevisioneffect.png",  floatFields = new (string, float)[0] },
            new SpecTemplate { componentName = "PixelationEffect",          outputName = "pixelation_default.png",  floatFields = new (string, float)[0] },
            new SpecTemplate { componentName = "myLed",                     outputName = "led_default.png",         floatFields = new (string, float)[0] },
            new SpecTemplate { componentName = "VortexEffect",              outputName = "vortex_default.png",      floatFields = new (string, float)[0] },
            // myWiggle default amplitude=0.01 is barely visible at 2048×1024.
            // Bump to 0.03 so the wave distortion is clearly readable in the
            // Unity capture and gives a meaningful Python comparison target.
            new SpecTemplate { componentName = "myWiggle",                  outputName = "wiggle_amp03.png",        floatFields = new[] { ("Amplitude", 0.03f) } },
            // myNoise at intensity=1.0 obliterates the scene with grey static
            // (Unity's literal formula `lerp(color, (color+n)/16, intensity)`
            // collapses image to 1/16 brightness at full intensity). Capture at
            // 0.25 instead so the scene is still recognisable AND noise is
            // clearly visible — gives a usable comparison target for Python.
            new SpecTemplate { componentName = "myNoise",                   outputName = "noise_int025.png",        floatFields = new[] { ("intensity", 0.25f), ("frequency", 0.5f) }, extraWaitSeconds = 1.5f },
            new SpecTemplate { componentName = "myGlitch",                  outputName = "glitch_default.png",      floatFields = new (string, float)[0] },
            // myTeichopsia uses a time-driven shimmer/mask; capturing at t≈0
            // tends to land on a frame where the mask is still ramping in.
            // Wait ~3s so the timer reaches a stable, visible "blue pixel mask"
            // phase — that's what Python apply_teichopsia should be matching.
            new SpecTemplate { componentName = "myTeichopsia",              outputName = "teichopsia_default.png",  floatFields = new (string, float)[0], extraWaitSeconds = 3.0f },
            // FovealDarkness defaults are tiny (innerCircleRadius=0.01, fadeWidth=0.05)
            // producing a barely-visible dot at the gaze point. Python
            // apply_foveal_darkness at sev=1.0 maps to radius~0.25 / fade~0.1 / full
            // black. Pin Unity to the same operating point.
            //
            // NOTE: the shader's `_Opacity` is INVERTED vs the field name.
            // Shader line 84: `lerp(black, color, _Opacity)` → _Opacity=1.0
            // means NO darkening (color shows through), _Opacity=0.0 means FULL
            // black. So for "max darkening" we set opacity=0.0.
            new SpecTemplate { componentName = "FovealDarkness",            outputName = "fovealdarkness_max.png",  floatFields = new[] { ("innerCircleRadius", 0.25f), ("fadeWidth", 0.1f), ("opacity", 0.0f) } },
            // myNystagmus has a foveation/rise window at the start where the
            // warp builds up; capturing at 0.5s lands inside that ramp and the
            // left part of the image still looks foveal-darkness-like. Wait
            // 2.5s so the oscillation has reached steady amplitude across the
            // whole frame.
            new SpecTemplate { componentName = "myNystagmus",               outputName = "nystagmus_amp20.png",     floatFields = new[] { ("artificialRotation", 1f), ("amp_deg", 20f), ("foveat_d", 1f), ("rise_d", 0.5f), ("rise_exp", 3f), ("screenWidth_px", 2048f), ("useNullingField", 0f) }, extraWaitSeconds = 2.5f },
            new SpecTemplate { componentName = "FlickeringStars",           outputName = "flickeringstars.png",     floatFields = new[] { ("numCoordinates", 15f), ("radius", 0.5f), ("starRadius", 0.02f), ("fadeInDuration", 0.1f), ("fadeOutDuration", 60f) }, extraWaitSeconds = 5.5f },
            new SpecTemplate { componentName = "myFloaters",                outputName = "floaters_dense.png",      floatFields = new[] { ("intensity", 1f), ("floaterSize", 2f), ("floaterDensity", 200f) } },
        };

        static int PopulateDefaultSpecs(MonoBehaviour capture, Type captureType, List<Type> attachedTypes)
        {
            var specsField = captureType.GetField("specs", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
            if (specsField == null) { Debug.LogWarning("[PixelFidelitySetup] Capture.specs field not found."); return 0; }
            var specType = captureType.GetNestedType("CaptureSpec");
            var ffType = captureType.GetNestedType("FloatField");
            if (specType == null || ffType == null) { Debug.LogWarning("[PixelFidelitySetup] CaptureSpec/FloatField nested types not found."); return 0; }

            var specsList = specsField.GetValue(capture) as System.Collections.IList;
            if (specsList == null) { Debug.LogWarning("[PixelFidelitySetup] Capture.specs is null."); return 0; }
            specsList.Clear();

            // Only add entries for filter types we actually attached.
            var attachedNames = new HashSet<string>(attachedTypes.Select(t => t.Name));
            int added = 0;
            foreach (var tmpl in DefaultSpecs)
            {
                if (!attachedNames.Contains(tmpl.componentName)) continue;
                var spec = Activator.CreateInstance(specType);
                specType.GetField("componentName").SetValue(spec, tmpl.componentName);
                specType.GetField("outputName").SetValue(spec, tmpl.outputName);

                var floatList = specType.GetField("floatFields").GetValue(spec) as System.Collections.IList;
                foreach (var ff in tmpl.floatFields)
                {
                    var ffInst = Activator.CreateInstance(ffType);
                    ffType.GetField("name").SetValue(ffInst, ff.name);
                    ffType.GetField("value").SetValue(ffInst, ff.value);
                    floatList.Add(ffInst);
                }
                var waitField = specType.GetField("extraWaitSeconds");
                if (waitField != null && tmpl.extraWaitSeconds > 0f)
                {
                    waitField.SetValue(spec, tmpl.extraWaitSeconds);
                }
                specsList.Add(spec);
                added++;
            }
            EditorUtility.SetDirty(capture);
            return added;
        }

        // --- helpers ------------------------------------------------------------

        static void EnsureTag(string tag)
        {
            var asset = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset");
            if (asset == null || asset.Length == 0) return;
            var so = new SerializedObject(asset[0]);
            var tagsProp = so.FindProperty("tags");
            for (int i = 0; i < tagsProp.arraySize; i++)
            {
                if (tagsProp.GetArrayElementAtIndex(i).stringValue == tag) return;
            }
            tagsProp.InsertArrayElementAtIndex(tagsProp.arraySize);
            tagsProp.GetArrayElementAtIndex(tagsProp.arraySize - 1).stringValue = tag;
            so.ApplyModifiedProperties();
        }

        static int RegisterAllShaders()
        {
            var graphicsAsset = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/GraphicsSettings.asset");
            if (graphicsAsset == null || graphicsAsset.Length == 0)
            {
                Debug.LogWarning("[PixelFidelitySetup] GraphicsSettings.asset not loadable.");
                return 0;
            }
            var so = new SerializedObject(graphicsAsset[0]);
            var arr = so.FindProperty("m_AlwaysIncludedShaders");

            var existing = new HashSet<Shader>();
            for (int i = 0; i < arr.arraySize; i++)
            {
                var s = arr.GetArrayElementAtIndex(i).objectReferenceValue as Shader;
                if (s != null) existing.Add(s);
            }

            if (!AssetDatabase.IsValidFolder(ShadersFolder))
            {
                Debug.LogWarning($"[PixelFidelitySetup] Shader folder '{ShadersFolder}' not found.");
                so.ApplyModifiedProperties();
                return existing.Count;
            }

            int added = 0;
            foreach (var guid in AssetDatabase.FindAssets("t:Shader", new[] { ShadersFolder }))
            {
                var path = AssetDatabase.GUIDToAssetPath(guid);
                var shader = AssetDatabase.LoadAssetAtPath<Shader>(path);
                if (shader == null || existing.Contains(shader)) continue;
                arr.arraySize++;
                arr.GetArrayElementAtIndex(arr.arraySize - 1).objectReferenceValue = shader;
                existing.Add(shader);
                added++;
            }
            so.ApplyModifiedProperties();
            return added;
        }

        static List<Type> DiscoverEffectTypes()
        {
            var baseType = FindType("VisSim.LinkableBaseEffect") ?? FindType("LinkableBaseEffect");
            var result = new List<Type>();
            if (baseType == null)
            {
                Debug.LogWarning("[PixelFidelitySetup] LinkableBaseEffect not found in any loaded assembly.");
                return result;
            }
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                Type[] types;
                try { types = asm.GetTypes(); }
                catch (System.Reflection.ReflectionTypeLoadException ex) { types = ex.Types.Where(t => t != null).ToArray(); }
                catch { continue; }
                foreach (var t in types)
                {
                    if (t == null) continue;
                    if (!t.IsAbstract && baseType.IsAssignableFrom(t) && t != baseType)
                        result.Add(t);
                }
            }
            return result;
        }

        static Type FindType(string nameOrFull)
        {
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                var t = asm.GetType(nameOrFull);
                if (t != null) return t;
            }
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                Type[] types;
                try { types = asm.GetTypes(); }
                catch { continue; }
                foreach (var tt in types)
                {
                    if (tt == null) continue;
                    if (tt.Name == nameOrFull || tt.FullName == nameOrFull) return tt;
                }
            }
            return null;
        }

        static void AttachDisabledWithShader(GameObject go, Type t, Dictionary<string, Shader> shaderByName)
        {
            var c = go.AddComponent(t) as MonoBehaviour;
            if (c == null) return;
            c.enabled = false;

            // Resolve a shader for this component. Several name conventions
            // may apply because (a) myBlur.GetShaderName returns
            // "Hidden/VisSim/myBlur" but its shader file declares
            // "Hidden/myBlur", (b) MonoBehaviour-only filters
            // (FovealDarkness etc.) don't override GetShaderName at all.
            var typeName = t.Name;
            var candidates = new List<string>();
            var fromMethod = TryGetShaderName(c);
            if (!string.IsNullOrEmpty(fromMethod)) candidates.Add(fromMethod);
            if (!string.IsNullOrEmpty(fromMethod) && fromMethod.StartsWith("Hidden/VisSim/"))
                candidates.Add("Hidden/" + fromMethod.Substring("Hidden/VisSim/".Length));
            candidates.Add("Hidden/VisSim/" + typeName);
            candidates.Add("Hidden/" + typeName);
            candidates.Add(typeName);

            Shader resolved = null;
            foreach (var name in candidates)
            {
                if (shaderByName.TryGetValue(name, out var s) && s != null) { resolved = s; break; }
            }
            if (resolved == null)
            {
                Debug.LogWarning($"[PixelFidelitySetup] No shader found for {typeName} (tried: {string.Join(", ", candidates)}).");
                return;
            }

            // Assign to every public Shader-typed field. Covers BaseEffect.Shader
            // AND custom names like darknessShader / pixelationShader / vortexShader
            // / starShader on the MonoBehaviour-only filters.
            const System.Reflection.BindingFlags F = System.Reflection.BindingFlags.Public
                | System.Reflection.BindingFlags.Instance
                | System.Reflection.BindingFlags.FlattenHierarchy;
            foreach (var fi in t.GetFields(F))
            {
                if (fi.FieldType == typeof(Shader))
                    fi.SetValue(c, resolved);
            }
        }

        static string TryGetShaderName(MonoBehaviour c)
        {
            var t = c.GetType();
            const System.Reflection.BindingFlags F =
                System.Reflection.BindingFlags.Instance |
                System.Reflection.BindingFlags.NonPublic |
                System.Reflection.BindingFlags.Public |
                System.Reflection.BindingFlags.FlattenHierarchy;
            // GetShaderName is `protected virtual` on BaseEffect.
            var mi = t.GetMethod("GetShaderName", F);
            if (mi == null) return null;
            try { return mi.Invoke(c, null) as string; }
            catch { return null; }
        }

        static Dictionary<string, Shader> BuildShaderLookup()
        {
            var dict = new Dictionary<string, Shader>();
            if (!AssetDatabase.IsValidFolder(ShadersFolder)) return dict;
            foreach (var guid in AssetDatabase.FindAssets("t:Shader", new[] { ShadersFolder }))
            {
                var path = AssetDatabase.GUIDToAssetPath(guid);
                var shader = AssetDatabase.LoadAssetAtPath<Shader>(path);
                if (shader == null) continue;
                dict[shader.name] = shader;
            }
            return dict;
        }

        static void DeleteByName(string n)
        {
            var go = GameObject.Find(n);
            if (go != null) UnityEngine.Object.DestroyImmediate(go);
        }

        static void DeleteAllWithTag(string tag)
        {
            // GameObject.FindGameObjectsWithTag throws if the tag isn't defined.
            try
            {
                var found = GameObject.FindGameObjectsWithTag(tag);
                foreach (var go in found)
                {
                    if (go != null) UnityEngine.Object.DestroyImmediate(go);
                }
            }
            catch (UnityException) { /* tag not yet defined */ }
        }

        static void CreateInputQuadIfTexturePresent()
        {
            Texture2D tex = null;
            foreach (var p in InputTextureCandidates)
            {
                tex = AssetDatabase.LoadAssetAtPath<Texture2D>(p);
                if (tex != null) break;
            }
            if (tex == null)
            {
                Debug.LogWarning("[PixelFidelitySetup] Trafficscene_2048x1024.png not found at any candidate path. " +
                                 "Create the InputQuad manually or import the texture.");
                return;
            }
            var quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
            quad.name = "InputQuad";
            quad.transform.localScale = new Vector3(20f, 10f, 1f);
            var mat = new Material(Shader.Find("Unlit/Texture"));
            mat.mainTexture = tex;
            quad.GetComponent<Renderer>().sharedMaterial = mat;
        }
    }
}
#endif
