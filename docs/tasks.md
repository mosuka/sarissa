# 2025-11-21 ドキュメントセントリック VectorCollection タスクリスト

## 対象とする目的

- 既存の `VectorCollection` を他モジュールから利用できるレベルまで拡張し、ドキュメント単位のベクトル検索を安全かつ柔軟に呼び出せるようにする。
- 追加の周辺ツールを実装する前に、検索 API/クエリ表現とレジストリ/メタデータ処理を固める。

## 関連ファイル

- `src/vector/collection.rs` : Collection API、検索実装、WAL/manifest ロジック
- `src/vector/core/document.rs` : `DocumentVectors`/`StoredVector` 型とシリアライゼーション
- `src/vector/index/{flat,hnsw,ivf}/` : フィールド別インデクサー（`VectorFieldWriter/Reader`）
- `src/vector/engine.rs`, `src/vector/search.rs` : 既存ベクトル検索エントリーポイント
- `src/hybrid/engine.rs`, `src/hybrid/search/merger.rs` : ハイブリッド検索経路とスコアマージ処理
- `docs/vector_collection.md` : 設計ドキュメント、クエリ仕様
- `docs/reports/2025-11-21-doc-centric-search.md` : 最新の doc-centric 対応状況の調査結果
- `examples/vector_search.rs`, `examples/vector_collection*.rs` (要追加) : 利用例を配置予定

## タスク一覧（優先順位順）

1. **クエリモデルとスコアリングの拡張**
   - `VectorCollectionQuery` に `score_mode`, `overfetch`, `FieldSelector`, クエリごとのロール/埋め込み制約を追加。
   - 検索実装でクエリベクトル→対象フィールドの整合性チェック、`VectorScoreMode` 切り替え、フィールド重み + クエリ重みの合成を実装。
   - `docs/vector_collection.md` とユニットテスト (`vector::collection::tests`) を新仕様に合わせて更新。
   - ✅ (2025-11-21) 上記一式を実装済み。`WeightedSum`/`MaxSim` モード、`FieldSelector`、`overfetch` のサポートを確認。
2. **レジストリ/メタデータフィルタリング**
   - `DocumentVectorRegistry` にクエリ時のフィルタリング API を用意して、`metadata` や `FieldEntry` の属性で対象ドキュメント/フィールドを絞り込めるようにする。
   - `VectorCollection::search` から新しいフィルタ構造体を受け取り、WAL/スナップショット再構築時にも必要メタデータが保持されることを確認。
   - 代表的なフィルタ（例: 言語・セクション）を網羅するテストケースとドキュメント例を追記。
   - ✅ (2025-11-21) `VectorCollectionFilter`/`MetadataFilter` を導入し、レジストリ経由で doc/field メタデータをフィルタリング。`vector::collection::tests` にドキュメント/フィールドメタデータのフィルタテストを追加し、`docs/vector_collection.md` を更新。
3. **既存ベクトルエンジンからのアダプタ提供**
   - `vector::engine` や `vector::search` で利用している旧 API から `VectorCollection` を呼び出すアダプタ層を実装し、段階的に移行できるようにする。
   - 互換層でのログ/計測を仕込み、既存利用者が挙動を比較できる手段を確保。
   - サンプルコード (`examples/vector_search.rs`) を `VectorCollectionQuery` 直接利用に書き換え、CLI なしで挙動を確認できるようにする。
   - ✅ (2025-11-21) 互換アダプタ (`search_with_legacy_request`) を削除し、移行は `VectorCollectionQuery` を直接構築する形に一本化。
   - ✅ (2025-11-21) `examples/vector_search.rs` を doc-centric な `VectorCollection` 直呼び出しサンプルへ更新。
   - ✅ (2025-11-21) `vector::engine` / `hybrid::engine` が `VectorCollectionQuery` を受け取るよう更新し、内部でレガシー検索リクエストへ変換する互換層を内包。
4. **E2E テスト & サンプルデータ**
   - 複数フィールド・複数ロールを含むサンプル `DocumentVectors` を `resources/` に追加し、`cargo test --package platypus --test collection_scenarios` のような E2E を用意。
   - `README.md` または `docs/vector_collection.md` にチュートリアル形式のガイドを記載。
   - ✅ (2025-11-22) `resources/vector_collection_sample.json` を追加し、`tests/vector_collection_scenarios.rs` で MemoryStorage 上の E2E クエリを検証。`README.md` と `docs/vector_collection.md` にサンプルデータ/テストへの導線を追記。

## 推奨される次の作業（2025-11-21 調査反映）

1. **VectorEngine 互換層の拡張**（`src/vector/engine.rs`）
   - 複数 `QueryVector` と複数 `FieldSelector` (Exact/Prefix/Role) を解析し、必要に応じてフィールドごとの `VectorSearchRequest` を組み立てる。
   - `VectorCollectionFilter` をサポートするまでの暫定策として、検索前にレジストリ経由のフィルタリング、または検索後に CPU フィルタを追加しメタデータ条件を満たさないヒットを除外する。
   - `VectorScoreMode::WeightedSum` / `MaxSim` で `VectorFieldConfig.base_weight`、`FieldVectors.weight`、`QueryVector.weight` を反映するスコア合成ロジックを doc-centric 仕様に合わせて更新する。
   - `LateInteraction` は非対応のままで良いが、ユーザに分かるエラーを整備し、今後の実装ポイントをコメントで明示する。
   - ✅ (2025-11-22) `src/vector/index/field.rs` で `FieldVectors.weight` をメタデータに反映し、`src/vector/engine.rs` および `vector::engine` テストでドキュメントメタデータフィルタをサポート。

2. **HybridEngine / ResultMerger の doc-centric 化**（`src/hybrid/engine.rs`, `src/hybrid/search/merger.rs`）
   - `HybridSearchRequest` に VectorCollection 専用パラメータ（フィールド指定、score_mode、フィルタ、overfetch 係数など）を追加し、`HybridEngine::build_vector_collection_query` で反映する。
   - Vector 検索結果の `field_hits` / メタデータを `ResultMerger` で参照してスコア加重およびレスポンス整形に利用する。特にハイブリッド重み調整時にフィールド別スコアを露出できるようにする。
   - Hybrid 結果のレスポンスモデルを更新し、VectorCollection 由来のメタデータ（例: フィールド名、ロール）を UI/クライアントが扱えるようにする。

3. **E2E テストとドキュメント整備**
   - ✅ `resources/vector_collection_sample.json` と `tests/vector_collection_scenarios.rs` を追加済み。今後は HybridEngine/CLI を含むエンドツーエンドシナリオを拡張し、VectorEngine/HybridEngine 双方で doc-centric パラメータを渡す動線を整える。
   - `docs/vector_collection.md` と README に Hybrid 経路からの `VectorCollectionQuery` 利用例、`MetadataFilter` の使い方、および score_mode の選択指針を追記する（README は導線のみ追加済み、Hybrid 具体例が未記載）。
   - CLI や example コードからも doc-centric パラメータを渡せるよう整備し、テスト結果を `docs/reports/` に追記して追跡可能にする。

## VectorCollection 直接移行計画（2025-11-21 更新）

互換アダプタを削除したため、今後は `VectorCollectionQuery` を正規の検索インターフェースとして統一する。以下の観点で段階的にモジュールを更新していく。

### 対象ファイル

- `src/vector/collection.rs` : Doc-centric API のみを公開し、不要な旧構造が混入しないよう継続的に監査。
- `src/vector/engine.rs` / `src/hybrid/` : 旧 `VectorSearchRequest` を直接 `VectorCollectionQuery` に置き換える実装計画を策定。
- `examples/vector_search.rs` : 最新の `VectorCollectionQuery` 作成方法を示すリファレンス（今回更新済み）。
- `docs/vector_collection.md` / `README.md` : ドキュメントを新 API ベースで保守し、旧 API への言及を削減。

### 作業ステップ

1. **呼び出し元の整理**
   - ✅ `vector::engine` と `hybrid::engine` が `VectorCollectionQuery` を直接構築・受け取るよう移行済み。今後は `vector::search` レイヤー（CLI/ユーティリティ）も同様に置換する。
2. **ドキュメントとサンプルの同期**
   - ✅ `README.md` の主要スニペットと `examples/vector_search.rs` を最新 API に更新済み。
   - `docs/vector_collection.md` へクエリ構築チュートリアルを追記し、ハイブリッド系ドキュメントとの差分を解消する。
3. **テスト/回帰チェック**
   - 既存の `vector::engine` 単体テストを `VectorCollectionSearchResults` ベースへ更新済み。引き続き `hybrid::engine` 向けの統合テストを追加し、互換層のロジックをカバーする。
