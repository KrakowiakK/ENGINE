# Third-Party Notices

This repository's own code is licensed under the MIT license in `LICENSE`. That
license covers the code that is not otherwise marked: `Sources/`, `Tests/`,
`tools/` and `apps/`. Parts of this repository started from Layr Labs' mlxfast
challenge starter (`Layr-Labs/qwen-3.8-mtp-challenge`), which is MIT licensed;
its notice is kept in `LICENSE.mlxfast-challenge`. The vendored libraries under
`Vendor/` keep their own licenses (`Vendor/mlx-swift/LICENSE`,
`Vendor/mlx-swift-lm/LICENSE` and the license files inside
`Vendor/mlx-swift/Source/Cmlx/`). The repository also depends on Swift packages
that SwiftPM downloads at build time. All of these are credited below under
their own licenses. **No model weights are distributed in this repository.**
The E9 checkpoint is published separately on Hugging Face under the Qwen
Community License 1.0 of its base model; the MIT license does not apply to it.

## Vendored source (shipped in this repository, modified)

Both vendored packages are forks of Apple's MLX Swift projects, taken through
Layr Labs' forks used by the mlxfast challenge, and modified further here
(additional Metal kernels, kernel dispatch changes, KV-cache and batching
support for the E9 engine). Each keeps its original license file.

| Path | Upstream | License |
|---|---|---|
| `Vendor/mlx-swift` | `ml-explore/mlx-swift` via `Layr-Labs/mlx-swift` | MIT (© 2023 ml-explore) -- `Vendor/mlx-swift/LICENSE` |
| `Vendor/mlx-swift/Source/Cmlx/mlx` | `ml-explore/mlx` via `Layr-Labs/mlx` (the MLX C++ core and Metal kernels) | MIT (© 2023 Apple Inc.) -- `Vendor/mlx-swift/Source/Cmlx/mlx/LICENSE` |
| `Vendor/mlx-swift/Source/Cmlx/mlx-c` | `ml-explore/mlx-c` via `Layr-Labs/mlx-c` | MIT (© 2023 ml-explore) -- `Vendor/mlx-swift/Source/Cmlx/mlx-c/LICENSE` |
| `Vendor/mlx-swift/Source/Cmlx/metal-cpp` | Apple metal-cpp | Apache-2.0 -- `Vendor/mlx-swift/Source/Cmlx/metal-cpp/LICENSE.txt` |
| `Vendor/mlx-swift/Source/Cmlx/fmt` | `fmtlib/fmt` | MIT (© Victor Zverovich and {fmt} contributors) -- `Vendor/mlx-swift/Source/Cmlx/fmt/LICENSE` |
| `Vendor/mlx-swift/Source/Cmlx/json` | `nlohmann/json` | MIT (© Niels Lohmann) -- `Vendor/mlx-swift/Source/Cmlx/json/LICENSE.MIT` |
| `Vendor/mlx-swift-lm` | `ml-explore/mlx-swift-lm` via `Layr-Labs/mlx-swift-lm` | MIT (© 2024 ml-explore) -- `Vendor/mlx-swift-lm/LICENSE` |

Further third-party code inside the vendored MLX sources keeps its notice in
the file itself (PocketFFT and metal-cpp are also listed in
`Vendor/mlx-swift/ACKNOWLEDGMENTS.md` and
`Vendor/mlx-swift/Source/Cmlx/mlx/ACKNOWLEDGMENTS.md`): PocketFFT
(`mlx/3rdparty/pocketfft.h`, BSD-3-Clause), the V8-derived
`mlx/mlx/small_vector.h` (BSD-3-Clause, © 2018 the V8 project authors and
© 2025 Apple Inc.), `threadpool.h` (zlib, © 2012 Jakob Progsch, Václav Zeman),
`expm1f.h` (BSD) and `cexpf.h` (Apache-2.0), a second metal-cpp copy
(`include-framework/Metal.hpp`, Apache-2.0), and googletest under
`fmt/test/gtest` (BSD-3-Clause) with `fmt/doc/python-license.txt` (PSF).
Copies of some of these files, with the same notices, also live under
`Cmlx/include-framework/` (`mlx-small_vector.h`, `mlx-threadpool.h`) and
`Cmlx/mlx-generated/` (`metal/cexpf.h`, `metal/expm1f.h`, and
`unary_ops.cpp`, which embeds them). `Cmlx/mlx/python/src/small_vector.h` is
not such a copy: it is Apple's own file (© 2025 Apple Inc., covered by the MLX
license), the Python-binding type caster that includes `mlx/small_vector.h`.
ENGINE did not modify any of these files or their copies.

The vendored trees also carry upstream documentation, benchmark reports and
scripts as they were imported; paths in those files (for example
`/Users/gaj/...`, `/Users/runner/...`) belong to the upstream authors and CI.

The vendored `mlx-swift-lm` also carries upstream model-architecture code for
Poolside's Laguna XS 2.1 (`Libraries/MLXLLM/Models/Laguna.swift`, the DFlash
libraries, a DFlash conversion script, tests and `docs/laguna-dflash.md`),
under the package's MIT license. ENGINE's E9 serving path does not use it, and
**no Laguna weights are shipped**. The upstream test fixture derived from
Poolside's `config.json` (`Tests/MLXLMTests/Resources/dflash-laguna-xs-2.1-config.json`,
OpenMDW-1.1) is not included in this repository, and its resource entry was
removed from `Vendor/mlx-swift-lm/Package.swift`; no test here loads it.

Upstream attribution statement, kept as published: "Laguna XS 2.1 and Laguna
XS 2.1 NVFP4 MLX © Poolside, licensed OpenMDW-1.1
(<https://huggingface.co/poolside/Laguna-XS-2.1>). Historical affine MLX
target conversion by mlx-community." The Laguna XS 2.1 DFlash speculator is
likewise © Poolside, licensed OpenMDW-1.1.

## Challenge starter

Parts of this repository (the build tooling, notably
`tools/build-mlx-metallib.sh`, and the original package layout) started from
Layr Labs' mlxfast challenge starter, `Layr-Labs/qwen-3.8-mtp-challenge` at
commit `0863b06ac16e26e48fc06e97444095b00feb66d4`, MIT licensed
(© 2026 Layr Labs, Inc.). Its license text is kept verbatim in
`LICENSE.mlxfast-challenge`. The challenge harness itself, its benchmark
pipeline and its model artifacts are not part of this repository; the closing
paragraph of `LICENSE.mlxfast-challenge` describes that upstream challenge
repository (including the Poolside Laguna models its harness downloads), not
this one. Layr Labs' (Eigen Labs) own modifications carried in `Vendor/`
(continuous batching, MoE, DFlash; see `Vendor/mlx-swift-lm/fork.yaml`), many of
which carry "Copyright © 2026 Eigen Labs." headers, are MIT licensed, covered
by the notice in `LICENSE.mlxfast-challenge` and by the MIT notice of each
vendored package. One of them,
`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/Paged/pagedattention.metal`,
records in its header that its structure was re-derived from vLLM's
paged-attention kernels (Apache-2.0), the mistral.rs Metal port (MIT,
© 2024 Eric Buehler) and vllm-metal (Apache-2.0); that header is kept.

## Reference implementations

ENGINE's model port, `Sources/Qwen4Exp/Qwen4Exp.swift`, is a Swift/MLX
rewrite written against three reference implementations. No reference source
file is shipped in this repository: the header of `Qwen4Exp.swift` says they
are kept under `lab/reference/`, and `lab/` belongs to the project's research
tree, which is not part of this export.

- Hugging Face `transformers`, `modeling_qwen4_exp.py` (the authoritative
  definition of `qwen4_exp`): Copyright 2026 The Qwen Team and The HuggingFace
  Inc. team, Apache License 2.0 (text in the appendix below). Changes: the
  model was rewritten in Swift on MLX.
- `ml-explore/mlx-lm` pull request #1788 (the vectorised MLX form of
  `qwen4_exp`, which the port follows op for op): MIT, Copyright © 2023 Apple
  Inc.; the license text is the same as
  `Vendor/mlx-swift/Source/Cmlx/mlx/LICENSE`.
- `ddalcu/mlx-serve`, `qwen4_exp.zig` (the host-side n-gram id hashing, which
  the port's n-gram id math follows): MIT, Copyright (c) 2026 David Dalcu
  (<https://github.com/ddalcu/mlx-serve>).

## Swift package dependencies (downloaded by SwiftPM, not vendored)

Pinned in `Package.resolved`; each license text is available in the package's
repository and, after a build, under `.build/checkouts/<package>/LICENSE*`:

| Package | License |
|---|---|
| `huggingface/swift-transformers`, `huggingface/swift-huggingface`, `huggingface/swift-jinja` | Apache-2.0 |
| Apple / swiftlang `swift-*` packages (`argument-parser`, `algorithms`, `asn1`, `async-algorithms`, `atomics`, `certificates`, `collections`, `configuration`, `crypto`, `distributed-tracing`, `http-structured-headers`, `http-types`, `log`, `metrics`, `nio` family, `numerics`, `service-context`, `syntax`, `system`) | Apache-2.0 |
| `swift-server/async-http-client`, `swift-server/swift-service-lifecycle`, `hummingbird-project/hummingbird` | Apache-2.0 |
| `ibireme/yyjson` | MIT |
| `mattt/EventSource` | MIT |

`swift-crypto` and `swift-nio-ssl` embed a copy of BoringSSL
(`CCryptoBoringSSL`, `CNIOBoringSSL`) whose files carry their own notices, and
several of the Apache-2.0 packages ship `NOTICE.txt` files. Nothing more is
owed for this source-only release; a prebuilt binary distribution would need
to include those NOTICE files and the BoringSSL notices.

## Model (not in this repository)

ENGINE serves **E9**, an 8-bit MLX quantisation of `Qwen/Qwen3.8-Flash-Next`
(model_type `qwen4_exp`). The weights are not part of this repository; they are
published separately at
<https://huggingface.co/aniolekx/Qwen3.8-Flash-Next-E9-MLX-8bit> under the
**Qwen Community License 1.0** of the base model. Review that license on the
model card before downloading, using or redistributing the weights. The MIT
license of this repository does not apply to them.

## Optional tooling (not bundled)

- `cmake` and Xcode's Metal Toolchain build `mlx.metallib`
  (`tools/build-mlx-metallib.sh`).
- Engine Studio can optionally start a Cloudflare quick tunnel through a
  separately installed `cloudflared`, and can optionally read PDF attachments
  with `pypdf` from a local virtualenv.

---

## Appendix: Apache License, Version 2.0

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS
