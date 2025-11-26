# VectorEngine マルチモーダル対応調査レポート (2025-11-25)

## 1. 背景と課題

- 現状の `VectorEngine` は `upsert_document_payload` による自動埋め込みがテキスト専用。
- `FieldPayload` は `text_segments` のみを持つ構造で、画像・音声といった別モダリティを同じ UX で取り込めない。
- 画像から生成したベクトルを手動注入することは可能だが、ユーザ作業コストが高く、`CandleMultimodalEmbedder` などの実装を十分に活かせていない。

## 2. 追加要件の整理

- テキストと同様の UX で画像ペイロードを ingestion できる。
- マルチモーダルを使わない既存ユーザの挙動を壊さない。
- Feature flag (`embeddings-multimodal`) の有無を正しく扱い、ビルドを壊さない。
- Embedder 実装がテキスト／画像の両対応かどうかを判定できる API が必要。
- 今後の拡張（音声、動画など）にも伸ばしやすい設計が望ましい。

## 3. 設計パターン比較

### パターンA: FieldPayload に画像セグメントを直接追加

- **概要**: `FieldPayload` に `image_segments`（将来的には `audio_segments` 等）を追加し、`embed_field_payload` でテキストと同列に処理する。
- **影響範囲**: `src/vector/core/document.rs` の構造改変、`TextEmbedder` から `ImageEmbedder` へのダウンキャスト、ドキュメント／テスト更新など。
- **メリット**: 既存 API との乖離が小さく、利用者は従来の `FieldPayload` に画像セグメントを足すだけで済む。
- **デメリット/リスク**: `TextEmbedder` トレイトに画像依存が混入し、feature gate 分岐が増える。モダリティ追加ごとに `FieldPayload` を肥大化させる必要がある。

### パターンB: モダリティ別 FieldPayload（enum 化）

- **概要**: `FieldPayload` を Text/Image/Mixed などの enum に変更し、フィールド定義時にモダリティを宣言する。
- **メリット**: フィールドごとの最適化や設定バリデーションが明確。
- **デメリット**: 既存 API に対して破壊的変更となるため、互換層の設計が不可欠。MixedPayload の仕様策定が難航する恐れ。

### パターンC: SegmentPayload DSL（VectorEntry から改名）

- **概要**: `FieldPayload` を `segments: Vec<SegmentPayload>` に刷新し、セグメント単位で `source`（text/image/url/binary）、`role`、`weight`、`metadata` を保持する DSL を導入する。
- **影響範囲**:
  - `FieldPayload` 構造、`DocumentPayload` API、`VectorEngine::embed_field_payload` の map/reduce 実装刷新。
  - Embedder Registry に「どの `PayloadSource` に対応するか」を問い合わせる API を追加。
- **メリット**: モダリティを追加する際は `SegmentPayload` や `PayloadSource` の variant を増やすだけで済み、設計上もっとも柔軟。
- **デメリット**: 実装コストが最大。既存の `add_text_segment` などを `SegmentPayload` 生成にラップする互換層が必要。

## 4. 推奨プラン

1. **短期 (MVP)**: パターンAで画像セグメントを追加し、`embeddings-multimodal` feature 有効時のみビルドされるようガードしてリリース速度を優先。
2. **中期**: Field 設定に `ingest_kind: TextOnly | ImageOnly | Multimodal` を加え、モダリティ別検証を行えるようにした上でパターンB/Cへの移行を計画。
3. **長期**: SegmentPayload DSL（パターンC）へ段階的にリファクタリングし、Query サイド（`VectorEngineSearchRequest` など）も同一 DSL で統一する。

## 5. 想定される追加タスク

- `FieldPayload` のシリアライザ/デシリアライザ更新と既存データ互換性チェック。
- `embedder_registry` のテスト（画像サポートなし embedder を使った際のエラー伝播）。
- `docs/` / README の更新、マルチモーダル使用例の追記。
- テスト: feature フラグ有無、画像セグメント混在、マルチモーダル embedder 未登録時のエラーなど。
- パフォーマンス: 画像処理は CPU/GPU を長時間占有するため、`EmbedderExecutor` のスレッド数やジョブキュー設計を見直す。

## 6. SegmentPayload DSL 概要

- `FieldPayload` は `segments: Vec<SegmentPayload>` を保持し、従来の `text_segments` は互換ラッパで `SegmentPayload` へ変換する。
- `SegmentPayload` は `source`, `vector_type`, `weight`, `metadata` を持ち、モダリティ・意味付け・重み付けをセグメント単位で制御できる。
- `PayloadSource` enum には `Text`, `ImageBytes`, `ImagePath`, `Url`, `ExternalVector` などを定義し、将来的に `AudioBytes` や `VideoFrame` を追加可能。
- `VectorEngine::embed_field_payload` は各セグメントごとに `VectorEmbedderRegistry::resolve(&segment.source)` を呼び出し、対応 embedder を実行する。
- Feature flag 無効時は `PayloadSource` ベースで compile-time/run-time の両面から防御し、非対応モダリティのセグメントを拒否する。
- Query サイドでも同じ DSL を使うことで、マルチモーダル検索 UX を統一できる。

### SegmentPayload / PayloadSource 型イメージ

```rust
pub struct FieldPayload {
    pub segments: Vec<SegmentPayload>,
    pub metadata: HashMap<String, String>,
}

pub struct SegmentPayload {
  pub source: PayloadSource,
  pub vector_type: VectorType,
  pub weight: f32,
  pub metadata: HashMap<String, String>,
}

pub enum PayloadSource {
  Text { value: String },
  Bytes { bytes: Arc<[u8]>, mime: MimeType },
  Uri { uri: String, media_hint: Option<MediaKind> },
  Vector { data: Arc<[f32]>, embedder_id: String },
}
```

### 実行・拡張ポイント

- SegmentPayload 追加 API（例: `add_text_segment_payload`, `add_image_segment_payload`）を `DocumentPayload` に提供し、既存メソッドから順次移行させる。
- `VectorFieldConfig` に `allowed_sources` や `default_role` を追加し、受付時のバリデーションを強化する。
- SegmentPayload に `priority` などの属性を追加すれば、画像など重い処理を伴うセグメントを別キューで制御できる。

## 7. (参考) レキシカル×ベクター ハイブリッド登録リクエスト例

> 実装詳細は本レポートのスコープ外だが、SegmentPayload DSL の利用イメージとして記載しておく。

- ドキュメント登録リクエストは `lexical` と `vector` の 2 ブロックで構成し、同一 `doc_id` でレキシカル／ベクターの両インデックスを同期させる。
- `vector` セクションは SegmentPayload DSL に相当する `content` 配列を持ち、セグメント単位で `source`・`type`（=`VectorType`）・`weight`・`metadata` を指定する。
- 複数のベクターフィールドを扱う場合は `vector` をマップ構造にし、フィールドごとに `content` を保持する。

```jsonc
{
  "doc_id": 12345,
  "metadata": {
    "tenant_id": "alpha"
  },
  "lexical": {
    "title": "マルチモーダル検索の最新動向",
    "body": "テキストと画像を組み合わせたハイブリッド検索が注目されています。"
  },
  "vector": {
    "metadata": {
      "lang": "ja"
    },
    "content": [
      {
        "source": {
          "type": "text",
          "value": "テキスト要約部分"
        },
        "type": "text",
        "weight": 1,
        "metadata": {
          "segment": "summary"
        }
      },
      {
        "source": {
          "type": "uri",
          "path": "file:///data/images/multimodal.png",
          "mime": "image/png"
        },
        "type": "image",
        "weight": 0.8,
        "metadata": {
          "segment": "hero_image"
        }
      }
    ]
  }
}
```

- 複数ベクターフィールド例（`body_embedding` と `product_image` を同時 upsert）:

```jsonc
{
  "doc_id": 67890,
  "lexical": {
    "title": "新製品レビュー",
    "body": "テキストと画像の両方で検索できるようにします。"
  },
  "vector": {
    "body_embedding": {
      "metadata": {
        "lang": "ja"
      },
      "content": [
        {
          "source": {
            "type": "text",
            "value": "本文の重要文"
          },
          "type": "text",
          "weight": 1,
          "metadata": {
            "section": "body"
          }
        }
      ]
    },
    "product_image": {
      "metadata": {
        "variant": "hero"
      },
      "content": [
        {
          "source": {
            "type": "uri",
            "path": "/assets/img/product.png",
            "mime": "image/png"
          },
          "type": "image",
          "weight": 0.9,
          "metadata": {
            "angle": "front"
          }
        }
      ]
    }
  }
}
```

このハイブリッド登録例はあくまで参考であり、詳細な実装計画は別ドキュメントで扱う想定とする。

---

本レポートでは、互換性重視の拡張 (パターンA) と将来拡張を見据えた設計 (パターンB/C) を比較した。まずはパターンAで画像 ingestion を早期解禁しつつ、SegmentPayload DSL をターゲットに段階的なリファクタリングを進めるのが現実的と考える。
