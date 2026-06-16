// WiggleBurstCapture.cs — STANDALONE burst-capture helper for the wiggle
// time-varying validation experiment. Modelled on FlickeringStarsBurstCapture.
//
// What it does (on Play):
//   1. Locates `captureCamera` (auto-find LeftEye by tag if not assigned).
//   2. Disables any PixelFidelityCapture in the scene to avoid conflicts.
//   3. Disables every VIP-Sim filter on the camera.
//   4. Enables only myWiggle and sets its params (AutomaticTimer=true so the
//      Timer field advances each frame). Defaults match Unity's authored
//      defaults: Frequency=12, Amplitude=0.01, Speed=1, Mode=Complex.
//   5. Waits `initialWaitSeconds`, then captures `burstFrames` frames at
//      `burstIntervalSeconds` intervals into `<outputFolder>/frame_NNN.png`.
//   6. Writes a `metadata.json` with per-frame `Timer` values (the only
//      input the wiggle shader uses besides static params).
//   7. Exits play mode (in editor) when done.
//
// Setup:
//   - Open the PixelFidelity scene (VipSim → Setup PixelFidelity Scene).
//   - GameObject → Create Empty → name "WiggleBurst".
//   - Add Component → WiggleBurstCapture.
//   - Optionally assign captureCamera (or leave blank to auto-find LeftEye).
//   - Press Play.

using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text;
using UnityEngine;

[DefaultExecutionOrder(1000)]
public class WiggleBurstCapture : MonoBehaviour
{
    [Header("Capture target")]
    public Camera captureCamera;
    public int outputWidth = 2048;
    public int outputHeight = 1024;

    [Tooltip("Folder relative to the project root (parent of Assets/).")]
    public string outputFolder = "Renders/wiggle_burst";

    [Header("Burst timing")]
    [Tooltip("Wait this long after Start() before capturing frame 0.")]
    public float initialWaitSeconds = 1.0f;

    [Tooltip("Total number of frames to capture.")]
    public int burstFrames = 30;

    [Tooltip("Seconds between captured frames. 0.5s * 30 = 15s of behaviour.")]
    public float burstIntervalSeconds = 0.5f;

    [Header("myWiggle params")]
    [Tooltip("0 = Simple, 1 = Complex (Unity default).")]
    public int mode = 1;
    public float speed = 1.0f;
    public float frequency = 12.0f;
    public float amplitude = 0.01f;

    private void Awake()
    {
        foreach (var pfc in FindObjectsByType<PixelFidelityCapture>(FindObjectsSortMode.None))
        {
            pfc.enabled = false;
            Debug.Log("[WiggleBurst] Disabled PixelFidelityCapture in scene to avoid conflict.");
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
            Debug.LogError("[WiggleBurst] captureCamera not assigned and no LeftEye tag found.");
            yield break;
        }

        string outDir = Path.GetFullPath(Path.Combine(Application.dataPath, "..", outputFolder));
        Directory.CreateDirectory(outDir);
        Debug.Log($"[WiggleBurst] Output dir: {outDir}");

        const BindingFlags F = BindingFlags.Public | BindingFlags.Instance | BindingFlags.NonPublic;
        MonoBehaviour wiggle = null;
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
            if (t.Name == "myWiggle") wiggle = mb;
        }
        if (wiggle == null)
        {
            Debug.LogError("[WiggleBurst] myWiggle component not found on captureCamera. Run VipSim -> Setup PixelFidelity Scene first.");
            yield break;
        }

        // Apply params BEFORE enabling so OnEnable picks them up.
        // myWiggle.Mode is an enum (Simple=0, Complex=1); SetField handles it.
        SetEnumField(wiggle, "Mode", mode, F);
        SetField(wiggle, "Speed", speed, F);
        SetField(wiggle, "Frequency", frequency, F);
        SetField(wiggle, "Amplitude", amplitude, F);
        SetField(wiggle, "AutomaticTimer", true, F);
        SetField(wiggle, "Timer", 0f, F);
        wiggle.enabled = true;
        Debug.Log($"[WiggleBurst] myWiggle enabled mode={mode} speed={speed} freq={frequency} amp={amplitude}.");

        yield return new WaitForEndOfFrame();
        yield return new WaitForEndOfFrame();
        if (initialWaitSeconds > 0f)
            yield return new WaitForSeconds(initialWaitSeconds);

        var times = new List<float>();
        var timers = new List<float>();
        float t0 = Time.time;
        var timerField = wiggle.GetType().GetField("Timer", F);

        for (int i = 0; i < burstFrames; i++)
        {
            yield return new WaitForEndOfFrame();
            float now = Time.time;
            float currentTimer = timerField != null ? (float)timerField.GetValue(wiggle) : 0f;
            times.Add(now - t0);
            timers.Add(currentTimer);
            string path = Path.Combine(outDir, $"frame_{i:D3}.png");
            SaveCameraToPNG(captureCamera, path);
            Debug.Log($"[WiggleBurst] [{i + 1}/{burstFrames}] t={now - t0:F2}s timer={currentTimer:F3} -> {Path.GetFileName(path)}");
            if (i < burstFrames - 1)
                yield return new WaitForSeconds(burstIntervalSeconds);
        }

        WriteMetadata(outDir, times, timers);
        Debug.Log("[WiggleBurst] DONE. Exiting play mode in 1s.");
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
            Debug.LogWarning($"[WiggleBurst] Field '{name}' not found on {comp.GetType().Name}.");
            return;
        }
        object v = value;
        if (fi.FieldType == typeof(int) && value is float f) v = Mathf.RoundToInt(f);
        fi.SetValue(comp, v);
    }

    private static void SetEnumField(MonoBehaviour comp, string name, int value, BindingFlags flags)
    {
        var fi = comp.GetType().GetField(name, flags);
        if (fi == null)
        {
            Debug.LogWarning($"[WiggleBurst] Enum field '{name}' not found on {comp.GetType().Name}.");
            return;
        }
        if (fi.FieldType.IsEnum)
        {
            fi.SetValue(comp, System.Enum.ToObject(fi.FieldType, value));
        }
        else
        {
            fi.SetValue(comp, value);
        }
    }

    private void WriteMetadata(string outDir, List<float> times, List<float> timers)
    {
        var inv = CultureInfo.InvariantCulture;
        string F(float v) => v.ToString("R", inv);
        var sb = new StringBuilder();
        sb.Append("{\n");
        sb.Append($"  \"filter\": \"wiggle\",\n");
        sb.Append($"  \"mode\": {mode},\n");
        sb.Append($"  \"speed\": {F(speed)},\n");
        sb.Append($"  \"frequency\": {F(frequency)},\n");
        sb.Append($"  \"amplitude\": {F(amplitude)},\n");
        sb.Append($"  \"initial_wait_seconds\": {F(initialWaitSeconds)},\n");
        sb.Append($"  \"burst_interval_seconds\": {F(burstIntervalSeconds)},\n");
        sb.Append($"  \"output_width\": {outputWidth},\n");
        sb.Append($"  \"output_height\": {outputHeight},\n");
        sb.Append("  \"frame_times_s\": [");
        for (int i = 0; i < times.Count; i++) { if (i > 0) sb.Append(", "); sb.Append(times[i].ToString("F4", inv)); }
        sb.Append("],\n");
        sb.Append("  \"timers\": [");
        for (int i = 0; i < timers.Count; i++) { if (i > 0) sb.Append(", "); sb.Append(timers[i].ToString("F6", inv)); }
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
            Debug.LogWarning($"[WiggleBurst] Write failed for {Path.GetFileName(path)}: {ex.Message}. Retrying via temp+rename.");
            string tmp = path + ".tmp";
            File.WriteAllBytes(tmp, bytes);
            try { if (File.Exists(path)) File.Delete(path); } catch { /* ignore */ }
            File.Move(tmp, path);
        }
        Object.Destroy(tex);
    }
}
