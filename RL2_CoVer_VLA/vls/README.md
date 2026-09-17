# VLS steering — port into RL2-VLA

Port of [VLS](https://arxiv.org/abs/2602.03973) (Liu et al. 2026, *Steering
Pretrained Robot Policies via Vision-Language Models*) into RL2-VLA, so VLS can
be run as a baseline on SIMPLER.

VLS steers a **frozen** pretrained diffusion/flow policy at inference time using
VLM-synthesized differentiable rewards over 3D keypoints — no retraining.

**Status:** adapter + perception layer (V1/V2) and guided sampler (V5) landed
and verified. Steering controller and rollout integration still to come.

**Scope:** WidowX / Bridge only. Google-Robot uses MEAN_STD action normalization
and a different control mode, so `SimplerAdapter` raises rather than silently
decoding trajectories wrongly.

---

## Environment

Use the **`rl2-vls`** conda env:

```bash
/home/user/.conda/envs/rl2-vls/bin/python
```

### Dependencies added for this port

Installed on top of the existing env; versions match VLS's own pins where they
exist (`VLS/requirements.txt`).

| Package | Version | Used by | Required? |
|---|---|---|---|
| `kmeans-pytorch` | 0.3 | `core/keypoint_detector.py` — clusters DINO features into keypoint candidates | yes |
| `parse` | 1.20.2 | `vlm_query/vlm_agent.py` — parses VLM responses | yes |
| `google-generativeai` | 0.8.6 | `core/gemini_grounder.py` — live stage recognition / replanning | optional (see below) |

```bash
/home/user/.conda/envs/rl2-vls/bin/pip install \
    "kmeans-pytorch==0.3" "parse==1.20.2" google-generativeai
```

> **protobuf caveat.** Installing `google-generativeai` pulls protobuf to 5.x,
> which violates TensorFlow 2.15's `<5.0.0` pin. Keep it at **`protobuf==4.25.9`**
> — plain `import tensorflow`, which is all the RL2 eval path uses
> (`simpler/eval_utils.py:25`, `simpler/simpler_utils.py:3`), works fine there.
>
> `tensorflow_datasets` is broken at *either* protobuf version because
> `tensorflow-metadata 1.21.0` ships protobuf gencode 6.31.1. That is a
> pre-existing conflict in the env, unrelated to this port, and `tfds` is not
> used by the eval path. Don't try to "fix" it by moving protobuf.

### API keys

| Variable | Needed for |
|---|---|
| `OPENAI_API_KEY` (+ optional `AZURE_OPENAI_BASE_URL`) | guidance-function synthesis |
| `GOOGLE_API_KEY` | live stage recognition (skip if running with cached guidance) |

Guidance functions can be generated once and replayed offline, which makes runs
deterministic and avoids API calls entirely.

---

## Verification

Scripts live in `vls/verify/` and run against a live SIMPLER env
(`widowx_carrot_on_plate`). Run them from that directory:

```bash
cd RL2_CoVer_VLA/vls/verify
/home/user/.conda/envs/rl2-vls/bin/python v1_v2_check.py          # perception: point cloud + segmentation
/home/user/.conda/envs/rl2-vls/bin/python v3_keypoint_check.py    # keypoint detection + tracking
/home/user/.conda/envs/rl2-vls/bin/python v3b_occlusion_check.py  # tracking is pose-based, not vision-based
/home/user/.conda/envs/rl2-vls/bin/python v5_gradient_check.py    # steering: decoder, gradients, FKD
/home/user/.conda/envs/rl2-vls/bin/python v6_obs_parity_check.py  # pi0 sees identical inputs vs CoVer
/home/user/.conda/envs/rl2-vls/bin/python make_visuals.py         # perception visual panel
```

Visuals land in `RL2-VLA/outputs/vls_verify/` (gitignored): `v1_v2_panel.png`,
`scene.ply`, `v3_tracking.png`, `v3b_occlusion.png`, `v5_guidance.png`.

### Does VLS change what the policy sees?

No — verified by `v6_obs_parity_check.py`. VLS switches the env to
`obs_mode="image"` to reach the raw `Position`/`Segmentation` textures it needs
for the world point cloud, but that does **not** alter pi0's inputs:

| Input | legacy (`rgbd`) vs VLS (`image`) |
|---|---|
| RGB | **bit-identical** — max abs diff 0 over 921,600 values |
| `eef_pos` | identical (0.0e+00) |
| `qpos` | identical (0.0e+00) |
| instruction | identical |

Both modes are views over the same render: the greenscreen overlay happens in
`base_env.py` before either path sees the data, and both apply the same uint8
conversion. `"image"` merely also exposes textures the `rgbd` wrapper discards.
This is what makes a VLS-vs-CoVer comparison measure the *steering* rather than
an observation change.

### The DINO feature extractor

**No API key is needed for keypoint detection.** `KeypointDetector` runs a local
DINO model and k-means-clusters its patch features inside each segmentation
mask. `OPENAI_API_KEY` is only used later to synthesize guidance functions;
`GOOGLE_API_KEY` only for live stage recognition.

VLS specifies `dinov3_vitb16`; we use **`dinov2_vitb14`**, which matches its
capacity (both 768-dim ViT-B). DINOv3 is unreachable here for two independent
reasons, neither fixed by relaxing pins:

1. It needs `transformers>=4.56`, and **4.56 breaks pi0** — `denoise_step()`
   raises `AttributeError: 'GemmaModel' object has no attribute 'model'`, since
   the lerobot fork depends on Gemma internals that changed. (Note `>=4.56.0`
   also resolves to 5.17.0, which is worse.) We stay on `transformers==4.48.3`.
2. In 4.48 the `DINOv3ViTModel` class doesn't exist at all, so there is nothing
   to load the weights into — ignoring the requirement doesn't help.

`torch.hub`'s `facebookresearch/dinov3` sidesteps transformers and loads, but
Meta ships only HF-format weights (211 tensors, `embeddings.*` naming) against
the hub model's 188 (`cls_token` naming), 2 keys overlapping — a conversion
would be unverifiable work on the critical path.

Because `keypoint_detector.py:148` silently falls back to `dinov2_vits14` on
**any** load failure, `make_detector()` raises if the loaded extractor family
differs from the one requested, rather than letting a substitution pass unnoticed.

Two detector settings also deviate from upstream, both from scene scale:

| Setting | VLS | Here | Why |
|---|---|---|---|
| `bounds_min/max` | CALVIN workspace | `z ∈ [0.80, 1.30]` | SIMPLER's table sits at z ≈ 0.87; CALVIN's box rejects every point |
| `min_dist_bt_keypoints` | 0.05 | 0.015 | Bridge objects are only ~11 cm across (carrot 11.3, plate 10.9), so a 5 cm merge radius collapses each object to a **single** keypoint |

### Reading `v5_guidance.png`

One decoded action chunk is 5 points (start + 4 steps) spanning ~7 cm. The left
panel shows two chunks from the same EE pose: **unguided** (purple) drifts up
and away, **guided** (yellow) turns down toward the carrot, ending 8.5 cm
closer. Guidance steers *direction* within a chunk — it does not reach a target
21 cm away in one chunk; the closed loop re-plans every `eff_act_steps`.

All three are self-contained (no arguments needed) and build their own SIMPLER
env via `_common.make_adapter()`, which also settles the scene past the reset
transient. Each check prints `PASS`/`FAIL` per line; a clean run is all `PASS`.
Expect ~1 min each, dominated by env startup.

`make_visuals.py` takes an optional output directory; it defaults to
`RL2-VLA/outputs/vls_verify/` (gitignored) and writes:

* `v1_v2_panel.png` — RGB / depth / segmentation / world-Z, with the TCP
  projected onto the first two panels
* `scene.ply` — world-frame coloured point cloud with object centroids marked
  in red (open in MeshLab or CloudCompare)

### What `v1_v2_check.py` asserts

| Check | Pass criterion | Why it matters |
|---|---|---|
| **V1a** reprojection round-trip | median < 1 px, p99 < 2 px | proves `cam2world_gl` (OpenGL) and `extrinsic_cv` (OpenCV) compose correctly — they are *different* conventions |
| **V1a′** mirror hypothesis | mirror error ≫ V1a error | rules out a vertical flip (LIBERO needs `flipud`; SIMPLER does not) |
| **V1b** TCP anchoring | gripper surface < 2 cm from TCP | validates the whole camera chain against independent ground truth. Measured against *finger* geometry: the TCP is the midpoint **between** the fingers, so the ray through its pixel passes into empty space |
| **V1c** dominant plane | < 2° from +Z, RMS < 5 mm | establishes that **world +Z is up** — the axis convention the guidance prompt template must state |
| **V1d** non-mutation | `Position` unchanged across calls | ManiSkill's reference point-cloud code mutates the obs array in place; the adapter must copy first |
| **V2b** robot leakage | no robot link ids in labelled segments | keypoints must never attach to the gripper, whose pose changes every step |
| **V2c** pose vs centroid | < 5 cm for compact actors | catches a mis-built `segment_index → entity` map, which would silently corrupt keypoint tracking |

### Why there is no V4 (decoder-vs-executed check)

The plan originally called for a V4 comparing the decoder's predicted trajectory
against the executed TCP path. That was needed while
`delta_actions_to_ee_trajectory` was an *independent reimplementation* of the
action→metres mapping, which could silently drift from what SIMPLER executes.

It no longer is. `_probe_action_affine()` measures the coefficients directly
from `convert_maniskill_with_bridge_adapter` — the exact function the rollout
applies before `env.step()` — at `f(0)`, `f(+1)`, `f(-1)`, and asserts the map
is genuinely affine. Agreement with the execution path is **4.8e-9 m**, by
construction rather than by measurement. V5a separately proves the Jacobian
exact to 5.24e-12.

What a V4 would still measure is residual physics: IK saturation, contact, joint
limits. Those are worth knowing but don't undermine steering here:

* the gradient is **unit-normalized**, so gain error is absorbed by `guide_scale`;
* no decoder models contact — including upstream VLS's;
* upstream never ran this check in CALVIN or LIBERO, with a *cruder* decoder
  (a single hardcoded scalar, no bias, no frame rotation).

If contact-phase behaviour ever looks suspect, the check to write is: replay
real pi0 chunks open-loop, and report the **cosine** between predicted and
achieved per-step displacement (not millimetres) split by free-space vs contact.
Do not drive the arm with random per-step jitter — the position controller
cannot track it and ~86% of steps produce zero motion, which measures a regime
the policy never operates in.

### What `v5_gradient_check.py` asserts

Covers the steering math. These gate everything downstream: a decoder with the
wrong frame or scale, or a gradient applied with the wrong sign, passes every
shape check while steering the robot confidently in the wrong direction.

| Check | Pass criterion | Why it matters |
|---|---|---|
| **V5a** decoder Jacobian | max rel err < 1e-8 (float64) | the decoder is affine, so finite differences must match the closed form `∂traj[k]/∂a[t,j] = R[:,j]·scale[j]` to machine precision. **Run in float64** — in float32 the same check reads ~2e-3 from cancellation noise, not a real error |
| **V5a2** denormalization bias | zero-action drift == `R·bias·T` | `denormalize_bound` is affine with a **nonzero bias** (~7 mm/step in z). A scale-only decoder silently loses ~2.8 cm per 4-step chunk |
| **V5b** reward gradient | cosine(finite-diff, autograd) > 0.999 | catches non-differentiable ops in VLM-written reward code. `load_functions_from_txt` validates *shape* but never gradient flow. Compares direction, since the returned gradient is unit-normalized |
| **V5c** composed update | reward rises **and** distance to target falls | see the sign note below — the single most dangerous silent failure |
| **V5d** FKD liveness | 11/11 unique timesteps, resample fires, particle 0 is max | regression test against re-introducing upstream pi05's `linspace` wiring, which `.long()`-truncates to `[1,0,0,…]` and silently disables FK steering entirely |
| **diversity** | works at B>1, returns `None` at B=1 | RBF repulsion self-disables at batch size 1, which is upstream's default |

> **Sign convention — three signs compose.** `autograd.grad` returns
> `+∂reward/∂sample` (the *ascent* direction); the velocity update *subtracts* it
> (`v' = v - scale·g`); and the flow-matching Euler step uses a **negative**
> `dt = -1/num_steps`. Net effect: `x' = x + dt·v + scale·|dt|·g`, so the sample
> moves *along* `+g` toward higher reward. Never test the gradient's sign in
> isolation — assert that the composed update raises reward and closes the
> distance to the target, which is what V5c does.

### Last recorded results (`widowx_carrot_on_plate`, seed 0)

```
V1a reprojection px err: median=0.7070 p99=0.7088           PASS
    mirror-hypothesis median err=250.00
V1b TCP anchor: finger px=5178 min|cloud-tcp|=0.0171 m      PASS
V1c dominant plane: normal=[0.003 0.001 1.0] 0.19°, 1.36mm  PASS
V1d Position unchanged=True, clouds identical=True          PASS
V2b robot-link leakage: none                                PASS
V2c carrot 0.0119 m | plate 0.0046 m                        PASS

V5a  Jacobian vs closed form (float64): 5.24e-12            PASS
V5a2 zero-action drift = [0.0008 0.0016 0.0279] m (2.80 cm) PASS
V5b  finite-diff vs autograd cosine = 1.000000              PASS
V5c  v -= s*g raises reward and closes distance             PASS
V5d  11/11 timesteps unique, resample fired 11/11,
     terminal sort puts best particle at index 0            PASS
```

---

## Layout

```
vls/
  core/
    fkd_class.py           Feynman-Kac particle resampling      (verbatim)
    keypoint_detector.py   DINO feature clustering -> keypoints (verbatim)
    keypoint_tracker.py    rigid-body keypoint attachment       (verbatim)
    gemini_grounder.py     Gemini grounding / stage recognition (verbatim)
    pi0_steer.py           guided sampler for RL2's pi0 policy  (NEW)
    env_adapters/
      base_adapter.py      abstract env contract                (verbatim + 1 fix)
      simpler_adapter.py   SIMPLER/SAPIEN implementation        (NEW)
  utils/                   logging, guidance loading, visualization (verbatim)
  vlm_query/               prompt templates + OpenAI agent      (verbatim)
  verify/
    _common.py             shared env/adapter setup
    v1_v2_check.py         perception checks
    v5_gradient_check.py   steering-math checks
    make_visuals.py        PNG panel + PLY point cloud
```

`pi0_steer.py` provides `GuidedSampler` (the steered flow-matching loop) and
`compute_guided_actions()`, a drop-in sibling of
`rl2_utils.compute_composed_actions` with an identical return signature, so the
rollout call site becomes a plain if/else.

Files marked *verbatim* are copied unchanged from upstream VLS, except that
imports are rewritten to the `vls.*` package so they resolve without requiring
VLS's repo root as the working directory. Any other deviation is marked with a
`# VLS-PORT:` comment explaining why, so a diff against upstream stays readable.
