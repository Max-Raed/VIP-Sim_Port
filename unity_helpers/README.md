# Unity helper — Pixel-Fidelity Capture

These two files set up a deterministic Unity scene that:
1. Loads `Trafficscene_2048x1024.png` (a power-of-two 2:1 input we both pipelines share)
2. Renders it through any chosen VIP-Sim filter at known parameters
3. Saves the result as a PNG named after the capture spec

After running once on your Windows PC, you copy the `Renders/` folder back to the Linux side and run `python scripts/abs_diff_filter.py …` to get the pixel-fidelity numbers.

---

## Files

- `PixelFidelityCapture.cs` — drop into `Assets/Scripts/` of the Unity project
- `Editor/PixelFidelitySceneSetup.cs` — drop into `Assets/Editor/` of the Unity project (the `Editor` folder name is required for Unity to compile it as editor-only). Adds a `VipSim → Setup PixelFidelity Scene` menu that builds the entire scene (LeftEye + RightEye + InputQuad + Capture) and attaches every VipSim filter component in one click.
- `../vipsim_assets/Trafficscene_2048x1024.png` — drop into `Assets/Resources/` of the Unity project

## Windows-side setup (one-time, ~10 min)

1. **Copy assets into the Unity project.** From WSL the path is
   `\\wsl.localhost\Ubuntu\home\mi-pool_8\acc-proj\avp-project\unity_helpers\PixelFidelityCapture.cs`
   and the Trafficscene PNG is at
   `\\wsl.localhost\Ubuntu\home\mi-pool_8\acc-proj\avp-project\vipsim_assets\Trafficscene_2048x1024.png`.

   In your Windows Unity project, copy:
   - `PixelFidelityCapture.cs` → `vipsim\windows\Assets\Scripts\`
   - `Trafficscene_2048x1024.png` → `vipsim\windows\Assets\Resources\Inputs\`
     (create the `Resources\Inputs\` subfolders if missing)

2. **Make a new scene.**
   - In Unity: `File → New Scene → Basic (URP) / 2D Built-in` (whichever appears in the dialog) → Create.
   - `File → Save As… → Assets/Scenes/PixelFidelity.unity`.

3. **Build the input quad.**
   - `GameObject → 3D Object → Quad`. Name it `InputQuad`.
   - Set its Transform → Scale to **(20, 10, 1)** (2:1 ratio at any size — the camera will frame it).
   - Position **(0, 0, 0)**.
   - In the Project window, click `Trafficscene_2048x1024` → in the Inspector set:
     - Texture Type: `Default`
     - Wrap Mode: `Clamp`
     - Filter Mode: `Bilinear`
     - Apply.
   - Drag `Trafficscene_2048x1024` from Project onto `InputQuad` in the Scene → it should appear textured.
   - Click `InputQuad` → its auto-generated Material is in `Assets/Materials/`. Open it. Set its Shader to `Unlit/Texture` (so no lighting interferes).

4. **Set up the capture camera.**
   - In the Scene's `Main Camera`, set:
     - Projection: `Orthographic`
     - Size: `5` (this exactly frames a 20×10 quad)
     - Position: `(0, 0, -10)`, Rotation `(0, 0, 0)`
     - Clear Flags: `Solid Color`, Background `(0,0,0,1)` (black, doesn't matter)
   - Verify in the Game view that you see the Trafficscene filling the viewport.

5. **Set Game view resolution to 2048 × 1024.**
   - At the top of the Game view, click the aspect-ratio dropdown → `+` → Type: `Fixed Resolution` → `2048 × 1024` → Add → select it.
   - (This isn't strictly required — the script renders to its own RenderTexture — but it makes preview accurate.)

6. **Add VIP-Sim filter components to the Main Camera.**
   For now, just add **myBlur** (we'll start with one filter to verify the loop):
   - Click `Main Camera` → Inspector → `Add Component` → search `myBlur` → Add.
   - Right-click on the new `My Blur` component header → `Set Component Disabled` (so it's off until our script enables it).

7. **Add the capture controller.**
   - `GameObject → Create Empty`. Name it `Capture`.
   - With `Capture` selected → `Add Component` → search `Pixel Fidelity Capture` → Add.
   - In the Inspector, fill in:
     - **Capture Camera:** drag `Main Camera` from the Hierarchy into this slot.
     - **Output Width:** `2048`, **Output Height:** `1024`
     - **Output Folder:** `Renders` (it will be created at `vipsim/windows/Renders/` after the first run)
     - **Specs:** click the `+` button to add **one** entry:
       - Component Name: `myBlur`
       - Output Name: `blur_kernal3.png`
       - Float Fields: click `+` → name `kernalSigma`, value `3`

8. **Press Play.**
   - Watch the Console.
   - Expected output:
     ```
     [PixelFidelity] Output dir: …/vipsim/windows/Renders
     [PixelFidelity] Effects found: 1
     [PixelFidelity] Saved baseline.png
     [PixelFidelity] [1/1] Saved blur_kernal3.png (myBlur)
     [PixelFidelity] DONE. Exiting play mode in 1s.
     ```
   - Play mode should auto-exit. Two PNGs sit in `vipsim\windows\Renders\`.

9. **Sanity check the PNGs.** Open `Renders/baseline.png` and `Renders/blur_kernal3.png` in any image viewer. The first should be the unmodified Trafficscene (cropped, 2048×1024), the second visibly blurred.

10. **Copy the renders back to Linux.** Easiest: drag `vipsim\windows\Renders\` folder to
    `\\wsl.localhost\Ubuntu\home\mi-pool_8\acc-proj\avp-project\vipsim_assets\unity_refs\`.

After this works for blur, we add more filter components + spec entries and re-press Play. One press of Play = one batch of all configured renders.

---

## Faster batch workflow (recommended after the blur smoke test)

Instead of manually adding each filter component, populating shader
fields, creating `LeftEye`/`RightEye` tags, and wiring up the capture
GameObject, use the auto-setup Editor menu:

1. **Copy** `Editor/PixelFidelitySceneSetup.cs` into
   `vipsim/windows/Assets/Editor/`. (Create the `Editor` folder if it
   doesn't exist — Unity treats any folder named `Editor` as editor-only.)
   Also make sure `PixelFidelityCapture.cs` is somewhere under
   `Assets/` (NOT under an `Editor/` folder). Unity will recompile.

2. **Open or create the scene** you want to use.
   File → New Scene (Basic Built-in) → Save As `Assets/Scenes/PixelFidelity.unity`.
   The script will build everything inside the active scene.

3. **Make sure the input texture is importable.**
   The script auto-creates an `InputQuad` if it can find the Trafficscene
   PNG at any of these paths:
   - `Assets/VisualEffects/Trafficscene_2048x1024.png`
   - `Assets/Trafficscene_2048x1024.png`
   - `Assets/vipsim_assets/Trafficscene_2048x1024.png`

   If yours is elsewhere, either copy it to one of those paths or edit
   `InputTextureCandidates` at the top of `PixelFidelitySceneSetup.cs`.

4. **Click `VipSim → Setup PixelFidelity Scene`.**
   The script will:
   - Ensure `LeftEye` and `RightEye` tags exist in TagManager.
   - Add every shader under `Assets/VisualEffects/Shaders/` to
     Project Settings → Graphics → Always Included Shaders. This fixes
     stripping of both primary and secondary `Hidden/VisSim/*` shaders
     (e.g. `cfxFrost` loaded by `myCataract`, `myScintillate` loaded by
     `myTeichopsia`).
   - Delete any existing `LeftEye`, `RightEye`, `InputQuad`, `Capture`
     in the scene (idempotent — safe to re-run).
   - Create `LeftEye` (orthographic Camera, tagged `LeftEye`),
     `RightEye` (disabled Camera sibling, tagged `RightEye`),
     `InputQuad` (20×10 unlit quad bound to Trafficscene), and
     `Capture` (with `PixelFidelityCapture`, `captureCamera` wired
     to the LeftEye camera).
   - Use reflection to discover every concrete `LinkableBaseEffect`
     subclass and `AddComponent` it on both eyes (disabled). Skips
     `myInpainter*` and `myFieldLossInverted`.

   Console output looks like:
   ```
   [PixelFidelitySetup] Done. Filters attached per eye: 19. Shaders
   registered as Always Included: 18. Save the scene (Ctrl+S) as
   PixelFidelity.unity.
   ```

5. **Save the scene** (`Ctrl+S`).

6. **Add capture specs.**
   Select the `Capture` GameObject and fill in the `Specs` list — one
   entry per (filter, parameters) render you want. Component name
   matches the C# type short name (e.g. `myBlur`, `myCataract`,
   `PixelationEffect`). Float fields are name + value pairs targeting
   public fields on that component (e.g. `maxCPD = 28`).

7. **Press Play.** One PNG per spec lands in
   `vipsim/windows/Renders/`, plus `baseline.png`.

8. **Copy the `Renders/` folder back to** `vipsim_assets/unity_refs/`.

That's the full Tier 2 batch workflow — one Editor click + spec list
edits + one Play press gets you Unity references for every filter.
