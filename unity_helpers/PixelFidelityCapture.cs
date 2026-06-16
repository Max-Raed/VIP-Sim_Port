// PixelFidelityCapture.cs — drop-in Unity helper for the AVP / VipSim
// pixel-fidelity validation pipeline.
//
// What it does (on Play):
//   1. Locates the configured capture Camera.
//   2. Disables all VIP-Sim filter components attached to it.
//   3. Saves a baseline render (no filter) as Renders/baseline.png.
//   4. For each entry in `specs`:
//        - finds the named MonoBehaviour component on the camera,
//        - sets its public fields per the override list,
//        - enables only it,
//        - renders one frame at outputWidth × outputHeight,
//        - saves the result as Renders/<outputName>.
//   5. Exits Play mode (in editor) when done.
//
// Supported field types in `floatFields`: float, int, bool, Vector3
// (encoded as "x,y,z" via the optional `vec3Fields` list).
//
// Setup (one-time):
//   - Add a Quad in scene that displays the input texture (Trafficscene_2048x1024).
//   - Add an Orthographic Camera framing the quad exactly. Camera renders to a
//     RenderTexture if you want pixel-exact output (recommended); the capture
//     here uses cam.Render() with a temporary RT, so any camera is fine.
//   - Add VIP-Sim filter components (myBlur, myCataract, …) to that Camera via
//     Add Component → search "VisSim".
//   - Add a new empty GameObject "Capture", attach this script, fill in
//     `captureCamera`, `outputFolder`, and the `specs` list.
//   - Press Play. Watch the Console for progress lines.

using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using UnityEngine;

[DefaultExecutionOrder(1000)]
public class PixelFidelityCapture : MonoBehaviour
{
    [Header("Capture")]
    public Camera captureCamera;
    public int outputWidth = 2048;
    public int outputHeight = 1024;

    [Tooltip("Folder relative to the project root (the parent of Assets/).")]
    public string outputFolder = "Renders";

    [Header("Specs — one entry per (filter, parameters) render")]
    public List<CaptureSpec> specs = new List<CaptureSpec>();

    [System.Serializable]
    public class FloatField
    {
        public string name;
        public float value;
    }

    [System.Serializable]
    public class Vec3Field
    {
        public string name;
        public Vector3 value;
    }

    [System.Serializable]
    public class CaptureSpec
    {
        [Tooltip("MonoBehaviour type name on the capture camera (e.g. 'myBlur', 'myCataract').")]
        public string componentName;

        [Tooltip("Output filename, e.g. 'blur_kernal3.png'.")]
        public string outputName;

        public List<FloatField> floatFields = new List<FloatField>();
        public List<Vec3Field> vec3Fields = new List<Vec3Field>();

        [Tooltip("Extra seconds to wait after enabling/applying fields, before capture. Use for time-dependent filters (async noise generation, fade-in, nystagmus cycle).")]
        public float extraWaitSeconds = 0f;
    }

    private List<MonoBehaviour> _allEffects = new List<MonoBehaviour>();

    private IEnumerator Start()
    {
        if (captureCamera == null)
        {
            Debug.LogError("[PixelFidelity] captureCamera not assigned.");
            yield break;
        }

        string outDir = Path.GetFullPath(Path.Combine(Application.dataPath, "..", outputFolder));
        Directory.CreateDirectory(outDir);
        Debug.Log($"[PixelFidelity] Output dir: {outDir}");

        // Collect candidate VIP-Sim effects on the camera. We grab anything in
        // the VisSim namespace plus a few common loose names (FovealDarkness etc.).
        foreach (var mb in captureCamera.GetComponents<MonoBehaviour>())
        {
            if (mb == null) continue;
            var t = mb.GetType();
            if (t.Namespace == "VisSim"
                || t.Name == "FovealDarkness"
                || t.Name == "myFloaters"
                || t.Name == "FlickeringStars"
                || t.Name == "PixelationEffect"
                || t.Name == "VortexEffect")
            {
                _allEffects.Add(mb);
            }
        }
        Debug.Log($"[PixelFidelity] Effects found: {_allEffects.Count}");

        DisableAll();
        yield return new WaitForEndOfFrame();
        yield return new WaitForEndOfFrame();

        // Baseline (no filter)
        SaveCameraToPNG(captureCamera, Path.Combine(outDir, "baseline.png"));
        Debug.Log("[PixelFidelity] Saved baseline.png");

        // Per spec
        for (int i = 0; i < specs.Count; i++)
        {
            var spec = specs[i];
            if (string.IsNullOrEmpty(spec.componentName) || string.IsNullOrEmpty(spec.outputName))
            {
                Debug.LogWarning($"[PixelFidelity] Spec #{i} missing componentName/outputName — skipped.");
                continue;
            }

            MonoBehaviour comp = FindEffect(spec.componentName);
            if (comp == null)
            {
                Debug.LogWarning($"[PixelFidelity] Component '{spec.componentName}' not found on capture camera — skipped (Add Component → VisSim → {spec.componentName}).");
                continue;
            }

            DisableAll();
            // Enable BEFORE applying fields: many VIP-Sim filters reset their
            // user-facing parameters inside OnEnable (e.g. myBlur sets maxCPD = 0
            // and recomputes kernalSigma). Applying fields after OnEnable means
            // OnRenderImage picks up the new values on the next frame.
            comp.enabled = true;
            yield return null;            // let OnEnable run
            ApplyFields(comp, spec);

            // Two end-of-frame waits: first for OnRenderImage to recompute with
            // the new field values, second to be safe against deferred work.
            yield return new WaitForEndOfFrame();
            yield return new WaitForEndOfFrame();

            // Optional extra wait for time-dependent filters (async texture
            // generation, fade-in animations, nystagmus saccade cycle, etc.).
            if (spec.extraWaitSeconds > 0f)
            {
                yield return new WaitForSeconds(spec.extraWaitSeconds);
                yield return new WaitForEndOfFrame();
            }

            SaveCameraToPNG(captureCamera, Path.Combine(outDir, spec.outputName));
            Debug.Log($"[PixelFidelity] [{i + 1}/{specs.Count}] Saved {spec.outputName} ({spec.componentName})");

            comp.enabled = false;
        }

        Debug.Log("[PixelFidelity] DONE. Exiting play mode in 1s.");
        yield return new WaitForSeconds(1f);

#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#endif
    }

    private void DisableAll()
    {
        foreach (var e in _allEffects) if (e != null) e.enabled = false;
    }

    private MonoBehaviour FindEffect(string nameOrFull)
    {
        foreach (var e in _allEffects)
        {
            var t = e.GetType();
            if (t.Name == nameOrFull) return e;
            if (t.FullName == nameOrFull) return e;
        }
        return null;
    }

    private void ApplyFields(MonoBehaviour comp, CaptureSpec spec)
    {
        var t = comp.GetType();
        const BindingFlags F = BindingFlags.Public | BindingFlags.Instance | BindingFlags.NonPublic;

        foreach (var ff in spec.floatFields)
        {
            if (string.IsNullOrEmpty(ff.name)) continue;
            var fi = t.GetField(ff.name, F);
            if (fi == null)
            {
                Debug.LogWarning($"[PixelFidelity] Field '{ff.name}' not found on {t.Name}.");
                continue;
            }
            object val = ff.value;
            if (fi.FieldType == typeof(int)) val = Mathf.RoundToInt(ff.value);
            else if (fi.FieldType == typeof(bool)) val = ff.value > 0.5f;
            else if (fi.FieldType == typeof(float)) val = ff.value;
            else if (fi.FieldType.IsEnum) val = System.Enum.ToObject(fi.FieldType, Mathf.RoundToInt(ff.value));
            else
            {
                Debug.LogWarning($"[PixelFidelity] Field '{ff.name}' on {t.Name} is {fi.FieldType.Name}; floatField cannot set it.");
                continue;
            }
            fi.SetValue(comp, val);
        }

        foreach (var vf in spec.vec3Fields)
        {
            if (string.IsNullOrEmpty(vf.name)) continue;
            var fi = t.GetField(vf.name, F);
            if (fi == null)
            {
                Debug.LogWarning($"[PixelFidelity] Field '{vf.name}' not found on {t.Name}.");
                continue;
            }
            if (fi.FieldType != typeof(Vector3))
            {
                Debug.LogWarning($"[PixelFidelity] Field '{vf.name}' on {t.Name} is {fi.FieldType.Name}; expected Vector3.");
                continue;
            }
            fi.SetValue(comp, vf.value);
        }
    }

    private void SaveCameraToPNG(Camera cam, string path)
    {
        var rt = RenderTexture.GetTemporary(outputWidth, outputHeight, 24, RenderTextureFormat.ARGB32);
        var prevTarget = cam.targetTexture;
        var prevActive = RenderTexture.active;

        cam.targetTexture = rt;
        cam.Render();

        RenderTexture.active = rt;
        var tex = new Texture2D(outputWidth, outputHeight, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, outputWidth, outputHeight), 0, 0);
        tex.Apply();

        cam.targetTexture = prevTarget;
        RenderTexture.active = prevActive;
        RenderTexture.ReleaseTemporary(rt);

        var bytes = tex.EncodeToPNG();
        // Delete-then-write to avoid Win32 IOException 1224 (file has a
        // user-mapped section open) caused by Explorer/Defender holding a
        // handle on a previously written PNG.
        try
        {
            if (File.Exists(path)) File.Delete(path);
            File.WriteAllBytes(path, bytes);
        }
        catch (IOException ex)
        {
            Debug.LogWarning($"[PixelFidelity] Write failed for {Path.GetFileName(path)}: {ex.Message}. Retrying via temp+rename.");
            string tmp = path + ".tmp";
            File.WriteAllBytes(tmp, bytes);
            try { if (File.Exists(path)) File.Delete(path); } catch { /* ignore */ }
            File.Move(tmp, path);
        }
        Object.Destroy(tex);
    }
}
