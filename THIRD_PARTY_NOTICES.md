# Third-Party Notices / 第三方声明

This file records source material bundled in, or directly adapted by, this
repository. Package dependencies installed at build or run time remain subject
to their own published licenses.

## GanymedeNil/document.ai

- Upstream: <https://github.com/GanymedeNil/document.ai>
- Audited revision: `b0a89f8bfb345ecac66cfd96af96b959b3be1908`
- Upstream author: GanymedeNil (`ganymedenil@gmail.com`)
- Upstream license: GNU Affero General Public License version 3; the upstream
  and repository `LICENSE` texts are byte-for-byte identical at the audited
  revision. The upstream repository does not separately state an SPDX
  `-only` / `-or-later` choice, so this notice does not infer one.
- Local paths and relationship:
  - `src/templates/index.html` is a verbatim copy of
    `code/server/templates/index.html` at the audited revision.
  - `src/ui_server.py` is adapted from `code/server/server.py`.
  - `src/load_words.py` is adapted from the ingestion design and code in
    `code/data_import/import_data.py`.

Copyright and license obligations for these paths remain with the upstream
author. Their inclusion does not grant e-dialect or Beijing Taju Technology
Co., Ltd. authority to offer them under alternative commercial terms.

---

本文件记录仓库内直接包含或改编的第三方来源；构建或运行时安装的依赖仍分别适用其
公开许可证。

上述路径包含对 `GanymedeNil/document.ai` 的原样复制或改编，继续适用上游作者的
版权与 GNU AGPL 第 3 版；上游未另行声明 SPDX 的 `-only` / `-or-later` 选择，
本声明不代为推定。收录这些内容不代表
e-dialect 或北京塔聚科技有限责任公司取得了以替代商业条款重许可这些内容的权利。
