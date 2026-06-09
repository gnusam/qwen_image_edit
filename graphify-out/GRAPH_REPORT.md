# Graph Report - qwen_image_edit_fork  (2026-06-09)

## Corpus Check
- 16 files · ~181,994 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 511 nodes · 666 edges · 60 communities (59 shown, 1 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `026c5b55`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]

## God Nodes (most connected - your core abstractions)
1. `inputs` - 11 edges
2. `inputs` - 11 edges
3. `inputs` - 11 edges
4. `inputs` - 11 edges
5. `inputs` - 11 edges
6. `input` - 8 edges
7. `input` - 8 edges
8. `input` - 7 edges
9. `handler()` - 7 edges
10. `inputs` - 7 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Communities (60 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.12
Nodes (24): apply_face_preservation(), check_cuda_availability(), download_file_from_url(), download_lora(), get_history(), get_image(), get_images(), handler() (+16 more)

### Community 1 - "Community 1"
Cohesion: 0.16
Nodes (19): inputs, inputs, inputs, inputs, inputs, cfg, denoise, latent_image (+11 more)

### Community 2 - "Community 2"
Cohesion: 0.11
Nodes (18): inputs, inputs, inputs, inputs, cfg, denoise, latent_image, lora_name (+10 more)

### Community 3 - "Community 3"
Cohesion: 0.11
Nodes (18): inputs, inputs, inputs, inputs, cfg, denoise, latent_image, lora_name (+10 more)

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (18): inputs, inputs, inputs, inputs, cfg, denoise, latent_image, lora_name (+10 more)

### Community 5 - "Community 5"
Cohesion: 0.15
Nodes (16): 3, class_type, _meta, 66, class_type, _meta, 75, class_type (+8 more)

### Community 6 - "Community 6"
Cohesion: 0.21
Nodes (12): _encode_file_to_base64(), get_config(), _get_s3_config(), main(), int, str, RunPod /runsync 호출. timeout(초)로 클라이언트 대기 시간과 서버 결과 유지 시간(wait) 설정., test.env 또는 환경변수에서 RunPod Network Volume S3 설정 읽기.     test.env 키(현재 리포): url, r (+4 more)

### Community 7 - "Community 7"
Cohesion: 0.14
Nodes (14): _meta, 3, class_type, _meta, 75, class_type, _meta, 89 (+6 more)

### Community 8 - "Community 8"
Cohesion: 0.18
Nodes (13): 202, class_type, inputs, _meta, inputs, 88, class_type, inputs (+5 more)

### Community 9 - "Community 9"
Cohesion: 0.17
Nodes (11): category, config, allowedCudaVersions, containerDiskInGb, gpuCount, gpuIds, runsOn, description (+3 more)

### Community 10 - "Community 10"
Cohesion: 0.17
Nodes (11): 205, class_type, 3, class_type, _meta, 78, class_type, _meta (+3 more)

### Community 11 - "Community 11"
Cohesion: 0.21
Nodes (12): 110, class_type, inputs, _meta, 111, class_type, inputs, _meta (+4 more)

### Community 12 - "Community 12"
Cohesion: 0.17
Nodes (11): 110, class_type, 111, class_type, _meta, 66, class_type, _meta (+3 more)

### Community 13 - "Community 13"
Cohesion: 0.24
Nodes (12): inputs, inputs, inputs, inputs, clip, image1, image2, image3 (+4 more)

### Community 14 - "Community 14"
Cohesion: 0.18
Nodes (11): _meta, 66, class_type, _meta, 75, class_type, _meta, 89 (+3 more)

### Community 15 - "Community 15"
Cohesion: 0.25
Nodes (11): inputs, inputs, inputs, inputs, clip, image1, image2, pixels (+3 more)

### Community 16 - "Community 16"
Cohesion: 0.24
Nodes (11): inputs, inputs, 78, class_type, inputs, _meta, inputs, image (+3 more)

### Community 17 - "Community 17"
Cohesion: 0.29
Nodes (11): 118, class_type, inputs, _meta, inputs, inputs, inputs, image (+3 more)

### Community 18 - "Community 18"
Cohesion: 0.22
Nodes (10): 110, class_type, inputs, _meta, 111, class_type, inputs, _meta (+2 more)

### Community 19 - "Community 19"
Cohesion: 0.22
Nodes (10): 203, class_type, inputs, _meta, 204, class_type, inputs, _meta (+2 more)

### Community 20 - "Community 20"
Cohesion: 0.20
Nodes (9): 110, class_type, _meta, 3, class_type, _meta, 8, class_type (+1 more)

### Community 21 - "Community 21"
Cohesion: 0.20
Nodes (10): 111, class_type, _meta, 66, class_type, _meta, 89, class_type (+2 more)

### Community 22 - "Community 22"
Cohesion: 0.22
Nodes (8): input, height, image_url, lora_scale, lora_url, prompt, seed, width

### Community 23 - "Community 23"
Cohesion: 0.22
Nodes (8): input, height, image_url, image_url_2, image_url_3, prompt, seed, width

### Community 24 - "Community 24"
Cohesion: 0.22
Nodes (9): 201, class_type, inputs, _meta, 39, class_type, inputs, _meta (+1 more)

### Community 25 - "Community 25"
Cohesion: 0.22
Nodes (9): inputs, 93, class_type, inputs, _meta, image, megapixels, resolution_steps (+1 more)

### Community 26 - "Community 26"
Cohesion: 0.50
Nodes (7): bool, main(), int, str, queue_prompt(), wait_for_comfy(), wait_for_prompt()

### Community 27 - "Community 27"
Cohesion: 0.25
Nodes (7): input, height, image_path, image_url_2, prompt, seed, width

### Community 28 - "Community 28"
Cohesion: 0.29
Nodes (7): 38, class_type, inputs, _meta, clip_name, device, type

### Community 29 - "Community 29"
Cohesion: 0.29
Nodes (7): 38, class_type, inputs, _meta, clip_name, device, type

### Community 30 - "Community 30"
Cohesion: 0.29
Nodes (7): 38, class_type, inputs, _meta, clip_name, device, type

### Community 31 - "Community 31"
Cohesion: 0.24
Nodes (7): 117, class_type, inputs, _meta, 119, class_type, _meta

### Community 32 - "Community 32"
Cohesion: 0.29
Nodes (7): 38, class_type, inputs, _meta, clip_name, device, type

### Community 33 - "Community 33"
Cohesion: 0.33
Nodes (6): 37, class_type, inputs, _meta, unet_name, weight_dtype

### Community 34 - "Community 34"
Cohesion: 0.33
Nodes (6): 60, class_type, inputs, _meta, filename_prefix, images

### Community 35 - "Community 35"
Cohesion: 0.33
Nodes (6): inputs, inputs, image, megapixels, resolution_steps, upscale_method

### Community 36 - "Community 36"
Cohesion: 0.33
Nodes (6): 37, class_type, inputs, _meta, unet_name, weight_dtype

### Community 37 - "Community 37"
Cohesion: 0.33
Nodes (6): 60, class_type, inputs, _meta, filename_prefix, images

### Community 38 - "Community 38"
Cohesion: 0.33
Nodes (6): 37, class_type, inputs, _meta, unet_name, weight_dtype

### Community 39 - "Community 39"
Cohesion: 0.33
Nodes (6): 60, class_type, inputs, _meta, filename_prefix, images

### Community 40 - "Community 40"
Cohesion: 0.33
Nodes (6): 37, class_type, inputs, _meta, unet_name, weight_dtype

### Community 41 - "Community 41"
Cohesion: 0.33
Nodes (6): 60, class_type, inputs, _meta, filename_prefix, images

### Community 42 - "Community 42"
Cohesion: 0.40
Nodes (4): config, allowedCudaVersions, gpuTypeIds, tests

### Community 43 - "Community 43"
Cohesion: 0.40
Nodes (5): 200, class_type, inputs, _meta, ckpt_name

### Community 44 - "Community 44"
Cohesion: 0.40
Nodes (5): 39, class_type, inputs, _meta, vae_name

### Community 45 - "Community 45"
Cohesion: 0.40
Nodes (5): 8, class_type, inputs, _meta, samples

### Community 46 - "Community 46"
Cohesion: 0.40
Nodes (5): 88, class_type, inputs, _meta, pixels

### Community 47 - "Community 47"
Cohesion: 0.40
Nodes (5): 39, class_type, inputs, _meta, vae_name

### Community 48 - "Community 48"
Cohesion: 0.40
Nodes (5): 39, class_type, inputs, _meta, vae_name

### Community 50 - "Community 50"
Cohesion: 0.50
Nodes (4): 78, class_type, inputs, _meta

### Community 51 - "Community 51"
Cohesion: 0.67
Nodes (3): 206, class_type, _meta

### Community 52 - "Community 52"
Cohesion: 0.67
Nodes (3): 8, class_type, _meta

### Community 53 - "Community 53"
Cohesion: 0.67
Nodes (3): 117, class_type, _meta

### Community 54 - "Community 54"
Cohesion: 0.67
Nodes (3): 118, class_type, _meta

### Community 55 - "Community 55"
Cohesion: 0.67
Nodes (3): 75, class_type, _meta

### Community 56 - "Community 56"
Cohesion: 0.67
Nodes (3): 88, class_type, _meta

### Community 57 - "Community 57"
Cohesion: 0.67
Nodes (3): 93, class_type, _meta

### Community 58 - "Community 58"
Cohesion: 0.67
Nodes (3): 120, class_type, _meta

### Community 59 - "Community 59"
Cohesion: 0.67
Nodes (3): 88, class_type, _meta

## Knowledge Gaps
- **190 isolated node(s):** `title`, `description`, `type`, `category`, `iconUrl` (+185 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `title` connect `Community 14` to `Community 33`, `Community 34`, `Community 8`, `Community 10`, `Community 43`, `Community 18`, `Community 19`, `Community 51`, `Community 52`, `Community 24`, `Community 28`?**
  _High betweenness centrality (0.009) - this node is a cross-community bridge._
- **Why does `3` connect `Community 7` to `Community 12`, `Community 4`?**
  _High betweenness centrality (0.008) - this node is a cross-community bridge._
- **Why does `inputs` connect `Community 4` to `Community 7`?**
  _High betweenness centrality (0.007) - this node is a cross-community bridge._
- **What connects `title`, `description`, `type` to the rest of the system?**
  _202 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.12333333333333334 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.1111111111111111 - nodes in this community are weakly interconnected._
- **Should `Community 3` be split into smaller, more focused modules?**
  _Cohesion score 0.1111111111111111 - nodes in this community are weakly interconnected._