# G0 Portable Conformance Manifest V1

`gov_g0_portable_conformance_manifest_v1.json` 是 canonical
`GOV-CJSON-1` index；其 `.sha256` sidecar 只包含 index 的 SHA-256 与文件名。
每个列出的 JSON page 也都是 canonical `GOV-CJSON-1` object。consumer 必须先
验证 index sidecar，再且只能按固定 `page_order` 读取 page，并在解析前核对每项
固定 basename、byte count 与 SHA-256。

当前 family 包含 13 个 canonical JSON page 与 8 个 raw parser input artifact。
wire page 包含 20 个内嵌 parser case、5 个 envelope case、2 个 host byte-adapter
case 与 3 个 public crypto transport case；schema pages 包含 4 个正向与 13 个反向
完整 envelope case，chain pages 各包含一个完整三集合 case。`u63-max-valid` 与
`u63-overflow` 是相邻的 signed-64 边界；正向 declaration vector 同时覆盖
sequence、两个 byte count 与 source_ref_key 的全部非固定位置为 max u63。

每个 page 都直接包含完整的 protocol input hex。schema case 使用一个完整
`input_envelope_hex`；chain case 使用完整的三组
`registry_envelopes_hex`、`revocation_envelopes_hex` 与
`declaration_envelopes_hex`。consumer 不得用 JSON encoder、Python object、
mutation operation 或 payload recipe 重建任何输入。

部分 parser boundary input 是 `raw_input_artifacts` 中列出的 raw `.bin` fixture。
这是 `GOV-CJSON-1` 的预算所必需：canonical object 的上限为 65,536 bytes，string
的上限为 4,096 bytes，而 4,096-byte string 与 65,536-byte document 边界的 direct
hex 无法放入一个 canonical string。`.bin` 是直接的 physical input bytes，不是
压缩数据，也不是 expansion format。consumer 只能在核对 index 中固定 basename、
byte count 与 SHA-256 后读取其精确 bytes。

该 index 是 portability conformance family，不是 signature verifier、root pin、
authorization mechanism、provenance qualification 或 capability grant。公开的
Ed25519 值只是不执行验签的 frozen transport vectors。
