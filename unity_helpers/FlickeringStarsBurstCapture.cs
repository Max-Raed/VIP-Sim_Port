// FlickeringStarsBurstCapture.cs — STANDALONE burst-capture helper for the
// flickering-stars time-varying validation experiment.
//
// This is INTENTIONALLY isolated from PixelFidelityCapture / PixelFidelitySceneSetup
// so that running it does NOT touch any of the 20-filter single-frame captures
// in vipsim_assets/unity_refs/. Different output folder, different component.
//
// What it does (on Play):
//   1. Locates `captureCamera` (auto-find LeftEye by tag if not assigned).
//   2. Disables any PixelFidelityCapture component in the scene so they don't fight.
//   3. Disables every VIP-Sim filter on the camera.
//   4. Enables only FlickeringStars and sets its params (15 stars, radius=0.5,
//      starRadius=0.02, fadeInDuration=0.1, fadeOutDuration=60). These match the
//      single-frame spec used in PixelFidelitySceneSetup.
//   5. Optionally `Random.InitState(seed)` for reproducibility.
//   6. Waits `initialWaitSeconds`, then captures `burstFrames` frames at
//      `burstIntervalSeconds` intervals into `<outputFolder>/frame_NNN.png`.
//   7. Writes a `metadata.json` with per-frame Time.time values.
//   8. Exits play mode (in editor) when done.
//
// Setup:
//   - Open the PixelFidelity scene (built by VipSim → Setup PixelFidelity Scene).
//   - GameObject → Create Empty → name "FlickeringStarsBurst".
//   - Add Component → FlickeringStarsBurstCapture.
//   - Assign captureCamera (or leave blank to auto-find by LeftEye tag).
//   - Press Play.

using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text;
using UnityEngine;

[DefaultExecutionOrder(1000)]
public class FlickeringStarsBurstCapture : MonoBehaviour
{
    [Header("Capture target")]
    public Camera captureCamera;
    public int outputWidth = 2048;
    public int outputHeight = 1024;

    [Tooltip("Folder relative to the project root (parent of Assets/).")]
    public string outputFolder = "Renders/flickeringstars_burst";

    [Header("Burst timing")]
    [Tooltip("Wait this long after Start() before capturing frame 0.")]
    public float initialWaitSeconds = 1.0f;

    [Tooltip("Total number of frames to capture.")]
    public int burstFrames = 30;

    [Tooltip("Seconds between captured frames. 0.5s × 30 = 15s of behaviour.")]
    public float burstIntervalSeconds = 0.5f;

    [Header("FlickeringStars params (matches single-frame spec)")]
    public int numCoordinates = 15;
    public float radius = 0.5f;
    public float starRadius = 0.02f;
    public float fadeInDuration = 0.1f;
    public float fadeOutDuration = 60f;

    [Header("Determinism")]
    [Tooltip("If >= 0, Random.InitState(seed) is called before FlickeringStars.Start runs.")]
    public int seed = 42;

    private void Awake()
    {
        // Disable any other capture script in the scene so we don't clobber each other.
        foreach (var pfc in FindObjectsByType<PixelFidelityCapture>(FindObjectsSortMode.None))
        {
            pfc.enabled = false;
            Debug.Log("[FSBurst] Disabled PixelFidelityCapture in scene to avoid conflict.");
        }

        if (seed >= 0)
        {
            Random.InitState(seed);
            Debug.Log($"[FSBurst] Random.InitState({seed}).");
        }
    }

    private IEnumerator Start()
    {
        if (captureCamera == null)
        {
            var go = GameObject.FindGameObjectWithTag("LeftEye");
            if (go != null) captureCamera = go.GetComponent<Camera>();
        }
        if (captureCamera == null)
        {
            Debug.LogError("[FSBurst] captureCamera not assigned and no LeftEye tag found in scene.");
            yield break;
        }

        string outDir = Path.GetFullPath(Path.Combine(Application.dataPath, "..", outputFolder));
        Directory.CreateDirectory(outDir);
        Debug.Log($"[FSBurst] Output dir: {outDir}");

        // Disable every VipSim filter on the camera, then enable only FlickeringStars.
        const BindingFlags F = BindingFlags.Public | BindingFlags.Instance | BindingFlags.NonPublic;
        MonoBehaviour flicker = null;
        foreach (var mb in captureCamera.GetComponents<MonoBehaviour>())
        {
            if (mb == null) continue;
            var t = mb.GetType();
            bool isVipSim = t.Namespace == "VisSim"
                || t.Name == "FovealDarkness"
                || t.Name == "myFloaters"
                || t.Name == "FlickeringStars"
                || t.Name == "PixelationEffect"
                || t.Name == "VortexEffect";
            if (isVipSim) mb.enabled = false;
            if (t.Name == "FlickeringStars") flicker = mb;
        }
        if (flicker == null)
        {
            Debug.LogError("[FSBurst] FlickeringStars component not found on captureCamera. Run VipSim → Setup PixelFidelity Scene first.");
            yield break;
        }

        // Apply our params BEFORE enabling so Start() picks them up.
        SetField(flicker, "numCoordinates", numCoordinates, F);
        SetField(flicker, "radius", radius, F);
        SetField(flicker, "starRadius", starRadius, F);
        SetField(flicker, "fadeInDuration", fadeInDuration, F);
        SetField(flicker, "fadeOutDuration", fadeOutDuration, F);
        flicker.enabled = true;
        Debug.Log($"[FSBurst] FlickeringStars enabled with n={numCoordinates}, r={radius}, sr={starRadius}, fi={fadeInDuration}, fo={fadeOutDuration}.");

        // Let Start() / first OnRenderImage run.
        yield return new WaitForEndOfFrame();
        yield return new WaitForEndOfFrame();

        if (initialWaitSeconds > 0f)
            yield return new WaitForSeconds(initialWaitSeconds);

        float t0 = Time.time;
        var times = new List<float>();
        for (int i = 0; i < burstFrames; i++)
        {
            yield return new WaitForEndOfFrame();
            float now = Time.time;
            times.Add(now - t0);
            string path = Path.Combine(outDir, $"frame_{i:D3}.png");
            SaveCameraToPNG(captureCamera, path);
            Debug.Log($"[FSBurst] [{i + 1}/{burstFrames}] t={now - t0:F2}s → {Path.GetFileName(path)}");
            if (i < burstFrames - 1)
                yield return new WaitForSeconds(burstIntervalSeconds);
        }

        WriteMetadata(outDir, times);
        Debug.Log("[FSBurst] DONE. Exiting play mode in 1s.");
        yield return new WaitForSeconds(1f);

#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#endif
    }

    private static void SetField(MonoBehaviour comp, string name, object value, BindingFlags flags)
    {
        var fi = comp.GetType().GetField(name, flags);
        if (fi == null)
        {
            Debug.LogWarning($"[FSBurst] Field '{name}' not found on {comp.GetType().Name}.");
            return;
        }
        object v = value;
        if (fi.FieldType == typeof(int) && value is float f) v = Mathf.RoundToInt(f);
        fi.SetValue(comp, v);
    }

    private void WriteMetadata(string outDir, List<float> times)
    {
        // Use InvariantCulture so floats are written with '.' not ',' (German locale fix).
        var inv = CultureInfo.InvariantCulture;
        string F(float v) => v.ToString("R", inv);
        var sb = new StringBuilder();
        sb.Append("{\n");
        sb.Append($"  \"seed\": {seed},\n");
        sb.Append($"  \"num_stars\": {numCoordinates},\n");
        sb.Append($"  \"radius\": {F(radius)},\n");
        sb.Append($"  \"star_radius\": {F(starRadius)},\n");
        sb.Append($"  \"fade_in_duration\": {F(fadeInDuration)},\n");
        sb.Append($"  \"fade_out_duration\": {F(fadeOutDuration)},\n");
        sb.Append($"  \"initial_wait_seconds\": {F(initialWaitSeconds)},\n");
        sb.Append($"  \"burst_interval_seconds\": {F(burstIntervalSeconds)},\n");
        sb.Append($"  \"output_width\": {outputWidth},\n");
        sb.Append($"  \"output_height\": {outputHeight},\n");
        sb.Append("  \"frame_times_s\": [");
        for (int i = 0; i < times.Count; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(times[i].ToString("F4", inv));
        }
        sb.Append("]\n}\n");
        File.WriteAllText(Path.Combine(outDir, "metadata.json"), sb.ToString());
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
        try
        {
            if (File.Exists(path)) File.Delete(path);
            File.WriteAllBytes(path, bytes);
        }
        catch (IOException ex)
        {
            Debug.LogWarning($"[FSBurst] Write failed for {Path.GetFileName(path)}: {ex.Message}. Retrying via temp+rename.");
            string tmp = path + ".tmp";
            File.WriteAllBytes(tmp, bytes);
            try { if (File.Exists(path)) File.Delete(path); } catch { /* ignore */ }
            File.Move(tmp, path);
        }
        Object.Destroy(tex);
    }
}
