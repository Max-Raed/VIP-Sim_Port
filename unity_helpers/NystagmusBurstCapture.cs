// NystagmusBurstCapture.cs — STANDALONE burst-capture helper for the nystagmus
// time-varying validation experiment. Modelled on FlickeringStarsBurstCapture.
//
// IMPORTANT: forces artificialRotation = true. The myNystagmus default mode
// rotates the camera transform, which (a) cannot be replayed in numpy and
// (b) leaves persistent rotation between play sessions. artificialRotation
// applies the saccade as a UV-shift in the shader, which is replayable.
//
// What it does (on Play):
//   1. Locates `captureCamera` (auto-find LeftEye by tag if not assigned).
//   2. Disables PixelFidelityCapture and every other VIP-Sim filter.
//   3. Enables only myNystagmus with artificialRotation=true,
//      useNullingField=false (otherwise gaze-contingent texture changes the
//      shift). Defaults match Carpenter & Burnett: foveat_d=0.12, rise_d=0.13,
//      rise_exp=1.75, amp_deg=8, baselineErr_deg=0.733.
//   4. Waits `initialWaitSeconds`, then captures `burstFrames` frames at
//      `burstIntervalSeconds` intervals into `<outputFolder>/frame_NNN.png`.
//   5. Writes `metadata.json` with per-frame timer_secs and the static
//      saccade params (Python re-simulates the cycle deterministically).
//   6. Exits play mode (in editor) when done.
//
// Setup:
//   - Open the PixelFidelity scene (VipSim -> Setup PixelFidelity Scene).
//   - GameObject -> Create Empty -> name "NystagmusBurst".
//   - Add Component -> NystagmusBurstCapture.
//   - Optionally assign captureCamera (or leave blank to auto-find LeftEye).
//   - Press Play.
//
// Note on the baseline-shift RNG: myNystagmus uses UnityEngine.Random for the
// per-cycle baseline jitter, which is impossible to bit-replay in numpy. We
// override baselineErr_deg = 0 in the capture so the Python and Unity sides
// match exactly. If you want the jittered version later, set baselineErr_deg
// in this script AND in the Python validator's --baseline-err flag.

using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text;
using UnityEngine;

[DefaultExecutionOrder(1000)]
public class NystagmusBurstCapture : MonoBehaviour
{
    [Header("Capture target")]
    public Camera captureCamera;
    public int outputWidth = 2048;
    public int outputHeight = 1024;

    [Tooltip("Folder relative to the project root (parent of Assets/).")]
    public string outputFolder = "Renders/nystagmus_burst";

    [Header("Burst timing")]
    [Tooltip("Wait this long after Start() before capturing frame 0.")]
    public float initialWaitSeconds = 1.0f;

    [Tooltip("Total number of frames to capture.")]
    public int burstFrames = 30;

    [Tooltip("Seconds between captured frames. Must be << (foveat_d + rise_d) " +
             "= 0.25s with defaults, otherwise all samples land at the same " +
             "cycle phase. 0.05s * 30 = 1.5s, ~6 cycles with 5 samples each.")]
    public float burstIntervalSeconds = 0.05f;

    [Header("myNystagmus params (Carpenter & Burnett defaults)")]
    public float foveat_d = 0.12f;
    public float rise_d = 0.13f;
    public float rise_exp = 1.75f;
    public float amp_deg = 8f;
    [Tooltip("Set to 0 for deterministic replay; >0 introduces UnityRNG noise that numpy cannot replay.")]
    public float baselineErr_deg = 0f;
    public float direction_deg = 0f;
    public int screenWidth_px = 2048;
    public float viewingAngle_deg = 100.0f;

    private void Awake()
    {
        foreach (var pfc in FindObjectsByType<PixelFidelityCapture>(FindObjectsSortMode.None))
        {
            pfc.enabled = false;
            Debug.Log("[NystBurst] Disabled PixelFidelityCapture in scene to avoid conflict.");
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
            Debug.LogError("[NystBurst] captureCamera not assigned and no LeftEye tag found.");
            yield break;
        }

        string outDir = Path.GetFullPath(Path.Combine(Application.dataPath, "..", outputFolder));
        Directory.CreateDirectory(outDir);
        Debug.Log($"[NystBurst] Output dir: {outDir}");

        const BindingFlags F = BindingFlags.Public | BindingFlags.Instance | BindingFlags.NonPublic;
        MonoBehaviour nyst = null;
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
            if (t.Name == "myNystagmus") nyst = mb;
        }
        if (nyst == null)
        {
            Debug.LogError("[NystBurst] myNystagmus component not found on captureCamera. Run VipSim -> Setup PixelFidelity Scene first.");
            yield break;
        }

        // Note on the prev_foveat_d / prev_rise_d / prev_rise_exp ladder in
        // myNystagmus.OnUpdate: the component flips foveat_d, rise_d, rise_exp
        // on first-detected-change (foveat_d = 1 - foveat_d, etc.). To get the
        // values we actually want at runtime, we pre-set the *inverted* values
        // here so the first OnUpdate flip-back lands on our target.
        SetField(nyst, "foveat_d", 1f - foveat_d, F);
        SetField(nyst, "rise_d", 1f - rise_d, F);
        SetField(nyst, "rise_exp", 4f - rise_exp, F);
        SetField(nyst, "amp_deg", amp_deg, F);
        SetField(nyst, "baselineErr_deg", baselineErr_deg, F);
        SetField(nyst, "direction_deg", direction_deg, F);
        SetField(nyst, "artificialRotation", true, F);
        SetField(nyst, "useNullingField", false, F);
        SetField(nyst, "screenWidth_px", screenWidth_px, F);
        SetField(nyst, "viewingAngle_deg", viewingAngle_deg, F);
        nyst.enabled = true;
        Debug.Log($"[NystBurst] myNystagmus enabled foveat={foveat_d} rise={rise_d} exp={rise_exp} amp={amp_deg}.");

        yield return new WaitForEndOfFrame();
        yield return new WaitForEndOfFrame();
        if (initialWaitSeconds > 0f)
            yield return new WaitForSeconds(initialWaitSeconds);

        var times = new List<float>();
        var timerSecs = new List<float>();
        float t0 = Time.time;
        var timerField = nyst.GetType().GetField("timer_secs", F);
        var shiftDegField = nyst.GetType().GetField("shift_deg", F);

        // --- one-time runtime parameter dump (diagnostic for Python parity) ---
        // Reads the actual runtime values of the saccade params and the geometry
        // fields, so we can confirm SetField via reflection + prev_* ladder
        // landed on the values we expect. If any of these differ from the
        // metadata.json the Python validator reads, that's the bug.
        object Get(string name) {
            var fi = nyst.GetType().GetField(name, F);
            return fi != null ? fi.GetValue(nyst) : "<missing>";
        }
        Debug.Log($"[NystBurst] runtime check: foveat_d={Get("foveat_d")} rise_d={Get("rise_d")} rise_exp={Get("rise_exp")} amp_deg={Get("amp_deg")} baselineErr_deg={Get("baselineErr_deg")} direction_deg={Get("direction_deg")} screenWidth_px={Get("screenWidth_px")} viewingAngle_deg={Get("viewingAngle_deg")} useNullingField={Get("useNullingField")} artificialRotation={Get("artificialRotation")}");

        // --- camera/RT geometry dump ---
        // Confirms what source.width OnRenderImage actually sees. If pixelWidth
        // is not 2048 (or differs from outputWidth), the shader normalizes
        // _Displace by a different denominator than Python assumes, which
        // multiplies the visible saccade shift by (pixelWidth/outputWidth).
        var preTarget = captureCamera.targetTexture;
        Debug.Log($"[NystBurst] camera check: pixelWidth={captureCamera.pixelWidth} pixelHeight={captureCamera.pixelHeight} pixelRect={captureCamera.pixelRect} rect={captureCamera.rect} targetTexture={(preTarget == null ? "null" : preTarget.width + "x" + preTarget.height)} stereoTargetEye={captureCamera.stereoTargetEye} allowHDR={captureCamera.allowHDR}");

        for (int i = 0; i < burstFrames; i++)
        {
            yield return new WaitForEndOfFrame();
            float now = Time.time;
            float currentTimer = timerField != null ? (float)timerField.GetValue(nyst) : 0f;
            float currentShift = shiftDegField != null ? (float)shiftDegField.GetValue(nyst) : 0f;
            times.Add(now - t0);
            timerSecs.Add(currentTimer);
            string path = Path.Combine(outDir, $"frame_{i:D3}.png");
            SaveCameraToPNG(captureCamera, path);
            Debug.Log($"[NystBurst] [{i + 1}/{burstFrames}] t={now - t0:F2}s timer={currentTimer:F3} shift_deg={currentShift:F3} -> {Path.GetFileName(path)}");
            if (i < burstFrames - 1)
                yield return new WaitForSeconds(burstIntervalSeconds);
        }

        WriteMetadata(outDir, times, timerSecs);
        Debug.Log("[NystBurst] DONE. Exiting play mode in 1s.");
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
            Debug.LogWarning($"[NystBurst] Field '{name}' not found on {comp.GetType().Name}.");
            return;
        }
        object v = value;
        if (fi.FieldType == typeof(int) && value is float f) v = Mathf.RoundToInt(f);
        fi.SetValue(comp, v);
    }

    private void WriteMetadata(string outDir, List<float> times, List<float> timerSecs)
    {
        var inv = CultureInfo.InvariantCulture;
        string F(float v) => v.ToString("R", inv);
        var sb = new StringBuilder();
        sb.Append("{\n");
        sb.Append($"  \"filter\": \"nystagmus\",\n");
        sb.Append($"  \"foveat_d\": {F(foveat_d)},\n");
        sb.Append($"  \"rise_d\": {F(rise_d)},\n");
        sb.Append($"  \"rise_exp\": {F(rise_exp)},\n");
        sb.Append($"  \"amp_deg\": {F(amp_deg)},\n");
        sb.Append($"  \"baselineErr_deg\": {F(baselineErr_deg)},\n");
        sb.Append($"  \"direction_deg\": {F(direction_deg)},\n");
        sb.Append($"  \"screenWidth_px\": {screenWidth_px},\n");
        sb.Append($"  \"viewingAngle_deg\": {F(viewingAngle_deg)},\n");
        sb.Append($"  \"artificialRotation\": true,\n");
        sb.Append($"  \"useNullingField\": false,\n");
        sb.Append($"  \"initial_wait_seconds\": {F(initialWaitSeconds)},\n");
        sb.Append($"  \"burst_interval_seconds\": {F(burstIntervalSeconds)},\n");
        sb.Append($"  \"output_width\": {outputWidth},\n");
        sb.Append($"  \"output_height\": {outputHeight},\n");
        sb.Append("  \"frame_times_s\": [");
        for (int i = 0; i < times.Count; i++) { if (i > 0) sb.Append(", "); sb.Append(times[i].ToString("F4", inv)); }
        sb.Append("],\n");
        sb.Append("  \"timer_secs\": [");
        for (int i = 0; i < timerSecs.Count; i++) { if (i > 0) sb.Append(", "); sb.Append(timerSecs[i].ToString("F6", inv)); }
        sb.Append("]\n}\n");
        File.WriteAllText(Path.Combine(outDir, "metadata.json"), sb.ToString());
    }

    private bool _renderLogged = false;
    private void SaveCameraToPNG(Camera cam, string path)
    {
        var rt = RenderTexture.GetTemporary(outputWidth, outputHeight, 24, RenderTextureFormat.ARGB32);
        var prevTarget = cam.targetTexture;
        var prevActive = RenderTexture.active;

        cam.targetTexture = rt;
        // one-shot log: print effective pixelWidth AFTER targetTexture is set,
        // because Unity sometimes re-derives pixelWidth from targetTexture rather
        // than from the camera's pixelRect.
        if (!_renderLogged)
        {
            Debug.Log($"[NystBurst] inside SaveCameraToPNG: rt={rt.width}x{rt.height} cam.pixelWidth={cam.pixelWidth} cam.pixelHeight={cam.pixelHeight} cam.targetTexture={(cam.targetTexture==null?"null":cam.targetTexture.width+"x"+cam.targetTexture.height)}");
            _renderLogged = true;
        }
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
            Debug.LogWarning($"[NystBurst] Write failed for {Path.GetFileName(path)}: {ex.Message}. Retrying via temp+rename.");
            string tmp = path + ".tmp";
            File.WriteAllBytes(tmp, bytes);
            try { if (File.Exists(path)) File.Delete(path); } catch { /* ignore */ }
            File.Move(tmp, path);
        }
        Object.Destroy(tex);
    }
}
