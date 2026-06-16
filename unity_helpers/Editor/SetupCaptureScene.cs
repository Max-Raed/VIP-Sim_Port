// SetupCaptureScene.cs — one-click batch setup for the pixel-fidelity scene.
//
// What it does (when you click `Tools/VipSim/Build Capture Scene`):
//   1. Discovers every concrete VIP-Sim filter MonoBehaviour:
//        - all non-abstract classes in the `VisSim` namespace whose base
//          chain includes BaseEffect (covers LinkableBaseEffect-derived
//          filters: myBlur, myCataract, myRecolour, myBCG, etc.)
//        - the loose ones in the global namespace: FovealDarkness,
//          FlickeringStars (mirrors PixelFidelityCapture's own pickup list).
//   2. For each type: AddComponent on the LeftEye GameObject AND on the
//      RightEye GameObject (both required for LinkableBaseEffect.OnEnable).
//      Components are left disabled — PixelFidelityCapture enables them
//      one at a time at runtime.
//   3. Walks AssetDatabase for every `Hidden/VisSim/*` shader and assigns
//      the matching one to the component's public `Shader` field (avoids
//      the "shader null" trap when the shader hasn't been referenced
//      anywhere else in the scene).
//   4. Replaces the `Capture` GameObject's `PixelFidelityCapture.specs`
//      list with one entry per filter (default parameters; user edits
//      values per filter as needed).
//
// Prerequisites (one-time):
//   - LeftEye tag on the Main Camera, RightEye tag on a sibling GameObject
//     with a (disabled) Camera component. Same as the manual workflow.
//   - GameObject named `Capture` with PixelFidelityCapture attached and
//     `captureCamera` assigned.
//
// Place this file at: vipsim/windows/Assets/Editor/SetupCaptureScene.cs
// (the `Editor` folder name is required so Unity compiles it as editor-only).

#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEngine;

public static class SetupCaptureScene
{
    private const string MENU = "Tools/VipSim/Build Capture Scene";

    [MenuItem(MENU)]
    public static void Build()
    {
        // 1. Find the Capture GameObject + its PixelFidelityCapture component
        var captureGO = GameObject.Find("Capture");
        if (captureGO == null)
        {
            EditorUtility.DisplayDialog("SetupCaptureScene",
                "No GameObject named 'Capture' in the scene.\n\n" +
                "Create an empty GameObject 'Capture' and attach the " +
                "PixelFidelityCapture script to it first.", "OK");
            return;
        }
        var capture = captureGO.GetComponent<PixelFidelityCapture>();
        if (capture == null)
        {
            EditorUtility.DisplayDialog("SetupCaptureScene",
                "'Capture' GameObject has no PixelFidelityCapture component.",
                "OK");
            return;
        }
        if (capture.captureCamera == null)
        {
            EditorUtility.DisplayDialog("SetupCaptureScene",
                "PixelFidelityCapture.captureCamera is not assigned.\n\n" +
                "Drag the LeftEye-tagged Main Camera into that field first.",
                "OK");
            return;
        }

        // 2. Locate LeftEye + RightEye GameObjects via tag
        var leftEye = GameObject.FindWithTag("LeftEye");
        var rightEye = GameObject.FindWithTag("RightEye");
        if (leftEye == null || rightEye == null)
        {
            EditorUtility.DisplayDialog("SetupCaptureScene",
                "Need GameObjects tagged 'LeftEye' and 'RightEye'.\n\n" +
                "Tag the Main Camera as LeftEye and create a sibling " +
                "GameObject (with a disabled Camera) tagged RightEye.",
                "OK");
            return;
        }

        // 3. Discover all VIP-Sim filter MonoBehaviour types via reflection
        var filterTypes = DiscoverFilterTypes();
        Debug.Log($"[SetupCaptureScene] Discovered {filterTypes.Count} filter types: " +
                  string.Join(", ", filterTypes.Select(t => t.Name)));

        // 4. Build a name → Shader lookup over every Hidden/VisSim/* shader
        var shaderByName = BuildShaderLookup();
        Debug.Log($"[SetupCaptureScene] Found {shaderByName.Count} 'Hidden/VisSim/*' shaders.");

        // 5. AddComponent + assign shader on both eyes; gather specs
        var newSpecs = new List<PixelFidelityCapture.CaptureSpec>();
        int added = 0, alreadyPresent = 0, shadersAssigned = 0;
        foreach (var t in filterTypes)
        {
            bool wasPresentLeft = leftEye.GetComponent(t) != null;
            bool wasPresentRight = rightEye.GetComponent(t) != null;

            var compL = leftEye.GetComponent(t)  ?? leftEye.AddComponent(t);
            var compR = rightEye.GetComponent(t) ?? rightEye.AddComponent(t);
            ((MonoBehaviour)compL).enabled = false;
            ((MonoBehaviour)compR).enabled = false;

            if (wasPresentLeft) alreadyPresent++; else added++;
            if (!wasPresentRight) added++;

            // Try to match a shader and assign to both
            string shaderName = TryGetShaderName(t, compL);
            if (shaderName != null && shaderByName.TryGetValue(shaderName, out var sh))
            {
                if (TryAssignShader(compL, sh) | TryAssignShader(compR, sh))
                    shadersAssigned++;
            }

            newSpecs.Add(new PixelFidelityCapture.CaptureSpec
            {
                componentName = t.Name,
                outputName = $"{t.Name}.png",
                floatFields = new List<PixelFidelityCapture.FloatField>(),
                vec3Fields = new List<PixelFidelityCapture.Vec3Field>(),
            });
        }

        capture.specs = newSpecs;
        EditorUtility.SetDirty(capture);
        EditorUtility.SetDirty(leftEye);
        EditorUtility.SetDirty(rightEye);
        UnityEditor.SceneManagement.EditorSceneManager.MarkSceneDirty(
            captureGO.scene);

        Debug.Log($"[SetupCaptureScene] DONE. " +
                  $"Filter types: {filterTypes.Count}. " +
                  $"Components added: {added} (already present: {alreadyPresent}). " +
                  $"Shaders assigned: {shadersAssigned}. " +
                  $"Specs populated: {newSpecs.Count}.");
        EditorUtility.DisplayDialog("SetupCaptureScene",
            $"Done. {filterTypes.Count} filters wired up.\n\n" +
            "Now: review the Capture GameObject's Specs list, edit per-filter " +
            "parameters as needed, save the scene, press Play.",
            "OK");
    }

    // --- discovery helpers ---

    private static List<Type> DiscoverFilterTypes()
    {
        var loose = new HashSet<string> { "FovealDarkness", "FlickeringStars" };
        var skip = new HashSet<string> { "myInpainter2_hacked" };
        var result = new List<Type>();

        foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
        {
            Type[] types;
            try { types = asm.GetTypes(); }
            catch (ReflectionTypeLoadException ex) { types = ex.Types.Where(x => x != null).ToArray(); }
            foreach (var t in types)
            {
                if (t == null || t.IsAbstract) continue;
                if (!typeof(MonoBehaviour).IsAssignableFrom(t)) continue;
                if (skip.Contains(t.Name)) continue;

                bool isVisSimEffect =
                    t.Namespace == "VisSim" && InheritsFrom(t, "BaseEffect");
                bool isLoose = loose.Contains(t.Name);
                if (isVisSimEffect || isLoose) result.Add(t);
            }
        }
        return result.OrderBy(t => t.Name).ToList();
    }

    private static bool InheritsFrom(Type t, string baseName)
    {
        var cur = t.BaseType;
        while (cur != null)
        {
            if (cur.Name == baseName) return true;
            cur = cur.BaseType;
        }
        return false;
    }

    private static Dictionary<string, Shader> BuildShaderLookup()
    {
        var dict = new Dictionary<string, Shader>();
        var guids = AssetDatabase.FindAssets("t:Shader");
        foreach (var guid in guids)
        {
            var path = AssetDatabase.GUIDToAssetPath(guid);
            var sh = AssetDatabase.LoadAssetAtPath<Shader>(path);
            if (sh != null && sh.name.StartsWith("Hidden/VisSim/"))
                dict[sh.name] = sh;
        }
        return dict;
    }

    private static string TryGetShaderName(Type t, UnityEngine.Object instance)
    {
        // GetShaderName() is protected in BaseEffect; use reflection.
        const BindingFlags F = BindingFlags.Public | BindingFlags.NonPublic |
                               BindingFlags.Instance | BindingFlags.FlattenHierarchy;
        var m = t.GetMethod("GetShaderName", F);
        if (m == null) return null;
        try { return m.Invoke(instance, null) as string; }
        catch { return null; }
    }

    private static bool TryAssignShader(UnityEngine.Object comp, Shader sh)
    {
        // BaseEffect declares `public Shader Shader;` (capital S).
        // Loose components (FovealDarkness etc.) declare their own Shader field.
        const BindingFlags F = BindingFlags.Public | BindingFlags.Instance;
        var fld = comp.GetType().GetField("Shader", F)
                  ?? comp.GetType().GetField("shader", F);
        if (fld == null || fld.FieldType != typeof(Shader)) return false;
        fld.SetValue(comp, sh);
        return true;
    }
}
#endif
