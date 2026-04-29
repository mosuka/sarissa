<?php

declare(strict_types=1);

use PHPUnit\Framework\TestCase;

/**
 * Basic integration tests for the laurus PHP binding.
 */
class LaurusTest extends TestCase
{
    // ── Helpers ──────────────────────────────────────────────────────────

    /**
     * Return a fresh in-memory index with two indexed documents.
     */
    private function createIndex(): Laurus\Index
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $schema->addTextField("body");
        $idx = new Laurus\Index(null, $schema);
        $idx->putDocument("doc1", ["title" => "Introduction to Rust", "body" => "Systems programming language."]);
        $idx->putDocument("doc2", ["title" => "Python for Data Science", "body" => "Data analysis with Python."]);
        $idx->commit();
        return $idx;
    }

    /**
     * Return an in-memory HNSW index with two indexed documents.
     */
    private function createVectorIndex(): Laurus\Index
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $schema->addHnswField("embedding", 4);
        $idx = new Laurus\Index(null, $schema);
        $idx->putDocument("doc1", ["title" => "Rust", "embedding" => [0.1, 0.2, 0.3, 0.4]]);
        $idx->putDocument("doc2", ["title" => "Python", "embedding" => [0.9, 0.8, 0.7, 0.6]]);
        $idx->commit();
        return $idx;
    }

    /**
     * Return an in-memory index with a Geo3d-typed `position` field.
     *
     * Coordinates are precomputed ECEF values (in meters) for well-known
     * landmarks, produced offline via `wgs84_to_ecef(lat, lon, height)`.
     */
    private function createGeo3dIndex(): Laurus\Index
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("name");
        $schema->addGeo3dField("position");
        $idx = new Laurus\Index(null, $schema);
        $idx->putDocument("tokyo_tower", [
            "name" => "Tokyo Tower",
            "position" => ["x" => -3955182.0, "y" => 3350553.0, "z" => 3700276.0],
        ]);
        $idx->putDocument("tokyo_skytree", [
            "name" => "Tokyo Skytree",
            "position" => ["x" => -3961178.0, "y" => 3346187.0, "z" => 3702490.0],
        ]);
        $idx->putDocument("mt_fuji", [
            "name" => "Mt. Fuji summit",
            "position" => ["x" => -3916073.0, "y" => 3437037.0, "z" => 3672751.0],
        ]);
        $idx->putDocument("sydney", [
            "name" => "Sydney Opera House",
            "position" => ["x" => -4646847.0, "y" => 2553022.0, "z" => -3534121.0],
        ]);
        $idx->commit();
        return $idx;
    }

    // ── Index creation ──────────────────────────────────────────────────

    public function testIndexMemory(): void
    {
        $idx = new Laurus\Index();
        $this->assertNotNull($idx);
    }

    public function testIndexWithSchema(): void
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $idx = new Laurus\Index(null, $schema);
        $this->assertNotNull($idx);
    }

    // ── Document CRUD ───────────────────────────────────────────────────

    public function testPutAndGetDocument(): void
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $idx = new Laurus\Index(null, $schema);
        $idx->putDocument("doc1", ["title" => "Hello"]);
        $idx->commit();
        $docs = $idx->getDocuments("doc1");
        $this->assertCount(1, $docs);
    }

    public function testPutReplacesExisting(): void
    {
        $idx = $this->createIndex();
        $idx->putDocument("doc1", ["title" => "Updated"]);
        $idx->commit();
        $docs = $idx->getDocuments("doc1");
        $this->assertCount(1, $docs);
    }

    public function testAddDocumentAppends(): void
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $idx = new Laurus\Index(null, $schema);
        $idx->addDocument("doc1", ["title" => "Chunk 1"]);
        $idx->addDocument("doc1", ["title" => "Chunk 2"]);
        $docs = $idx->getDocuments("doc1");
        $this->assertCount(2, $docs);
    }

    public function testDeleteDocuments(): void
    {
        $idx = $this->createIndex();
        $idx->deleteDocuments("doc1");
        $idx->commit();
        $docs = $idx->getDocuments("doc1");
        $this->assertCount(0, $docs);
    }

    public function testGetUnknownId(): void
    {
        $idx = $this->createIndex();
        $docs = $idx->getDocuments("unknown");
        $this->assertCount(0, $docs);
    }

    // ── Statistics ───────────────────────────────────────────────────────

    public function testDocumentCount(): void
    {
        $idx = $this->createIndex();
        $stats = $idx->stats();
        $this->assertEquals(2, $stats["document_count"]);
    }

    public function testVectorFieldStats(): void
    {
        $idx = $this->createVectorIndex();
        $stats = $idx->stats();
        $this->assertEquals(2, $stats["document_count"]);
        $this->assertArrayHasKey("embedding", $stats["vector_fields"]);
        $this->assertEquals(4, $stats["vector_fields"]["embedding"]["dimension"]);
    }

    // ── Lexical search ──────────────────────────────────────────────────

    public function testSearchDsl(): void
    {
        $idx = $this->createIndex();
        $results = $idx->search("title:rust");
        $this->assertCount(1, $results);
        $this->assertEquals("doc1", $results[0]->getId());
        $this->assertGreaterThan(0, $results[0]->getScore());
    }

    public function testSearchTermQuery(): void
    {
        $idx = $this->createIndex();
        $q = new Laurus\TermQuery("title", "python");
        $results = $idx->search($q);
        $this->assertCount(1, $results);
        $this->assertEquals("doc2", $results[0]->getId());
    }

    public function testSearchWithLimit(): void
    {
        $idx = $this->createIndex();
        $results = $idx->search("title:rust OR title:python", 1);
        $this->assertCount(1, $results);
    }

    public function testSearchWithOffset(): void
    {
        $idx = $this->createIndex();
        $all = $idx->search("title:rust OR title:python", 10);
        $offset = $idx->search("title:rust OR title:python", 10, 1);
        $this->assertCount(count($all) - 1, $offset);
    }

    public function testSearchNoResults(): void
    {
        $idx = $this->createIndex();
        $results = $idx->search("title:nonexistent");
        $this->assertCount(0, $results);
    }

    // ── Query types ─────────────────────────────────────────────────────

    public function testPhraseQuery(): void
    {
        $idx = $this->createIndex();
        $q = new Laurus\PhraseQuery("body", ["systems", "programming"]);
        $results = $idx->search($q);
        $this->assertCount(1, $results);
    }

    public function testFuzzyQuery(): void
    {
        $idx = $this->createIndex();
        $q = new Laurus\FuzzyQuery("title", "rast");
        $results = $idx->search($q);
        $this->assertCount(1, $results);
    }

    public function testBooleanQuery(): void
    {
        $idx = $this->createIndex();
        $bq = new Laurus\BooleanQuery();
        $bq->must(new Laurus\TermQuery("title", "rust"));
        $bq->mustNot(new Laurus\TermQuery("title", "python"));
        $results = $idx->search($bq);
        $this->assertCount(1, $results);
        $this->assertEquals("doc1", $results[0]->getId());
    }

    public function testWildcardQuery(): void
    {
        $idx = $this->createIndex();
        $q = new Laurus\WildcardQuery("title", "ru*");
        $results = $idx->search($q);
        $this->assertCount(1, $results);
    }

    public function testNumericRangeQuery(): void
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $schema->addIntegerField("year");
        $idx = new Laurus\Index(null, $schema);
        $idx->putDocument("d1", ["title" => "old", "year" => 2000]);
        $idx->putDocument("d2", ["title" => "new", "year" => 2024]);
        $idx->commit();
        $q = new Laurus\NumericRangeQuery("year", 2020, 2030);
        $results = $idx->search($q);
        $this->assertCount(1, $results);
        $this->assertEquals("d2", $results[0]->getId());
    }

    // ── SpanQuery ───────────────────────────────────────────────────────

    public function testSpanQueryNear(): void
    {
        // Plain term-only `near` already works via `SpanQuery::near`.
        $schema = new Laurus\Schema();
        $schema->addTextField("body", true, true, true);
        $idx = new Laurus\Index(null, $schema);
        $idx->putDocument("d1", ["body" => "the quick brown fox"]);
        $idx->putDocument("d2", ["body" => "the slow red fox"]);
        $idx->commit();
        $q = Laurus\SpanQuery::near("body", ["quick", "fox"], 2, true);
        $results = $idx->search($q);
        $this->assertCount(1, $results);
        $this->assertEquals("d1", $results[0]->getId());
    }

    public function testSpanQueryNearSpansWithNestedClauses(): void
    {
        // `nearSpans` accepts pre-built SpanQuery instances, so callers can
        // nest other span constructions (e.g. another `near`, `containing`,
        // `within`) inside a `SpanNear`. The simplest exercise wraps two
        // bare `term` clauses and verifies the result is identical to the
        // plain `near` form.
        $schema = new Laurus\Schema();
        $schema->addTextField("body", true, true, true);
        $idx = new Laurus\Index(null, $schema);
        $idx->putDocument("d1", ["body" => "the quick brown fox"]);
        $idx->putDocument("d2", ["body" => "the slow red fox"]);
        $idx->commit();
        $clauses = [
            Laurus\SpanQuery::term("body", "quick"),
            Laurus\SpanQuery::term("body", "fox"),
        ];
        $q = Laurus\SpanQuery::nearSpans("body", $clauses, 2, true);
        $results = $idx->search($q);
        $this->assertCount(1, $results);
        $this->assertEquals("d1", $results[0]->getId());
    }

    // ── Geo3d (3D ECEF) ─────────────────────────────────────────────────

    public function testGeo3dFieldRoundTrip(): void
    {
        $idx = $this->createGeo3dIndex();
        $docs = $idx->getDocuments("tokyo_tower");
        $this->assertCount(1, $docs);
        $this->assertEquals("Tokyo Tower", $docs[0]["name"]);
        $this->assertEqualsWithDelta(-3955182.0, $docs[0]["position"]["x"], 1.0);
        $this->assertEqualsWithDelta(3350553.0, $docs[0]["position"]["y"], 1.0);
        $this->assertEqualsWithDelta(3700276.0, $docs[0]["position"]["z"], 1.0);
    }

    public function testGeo3dDistanceQuerySmallRadius(): void
    {
        // 50 km sphere around Tokyo Tower returns Tower + Skytree only.
        $idx = $this->createGeo3dIndex();
        $q = Laurus\Geo3dDistanceQuery::withinSphere(
            "position", -3955182.0, 3350553.0, 3700276.0, 50000.0
        );
        $results = $idx->search($q, 10);
        $ids = array_map(fn($r) => $r->getId(), $results);
        sort($ids);
        $this->assertEquals(["tokyo_skytree", "tokyo_tower"], $ids);
    }

    public function testGeo3dBoundingBoxQuery(): void
    {
        // Central-Tokyo box returns Tower + Skytree only (Mt. Fuji and Sydney
        // are outside the small AABB).
        $idx = $this->createGeo3dIndex();
        $q = Laurus\Geo3dBoundingBoxQuery::withinBox(
            "position",
            -3962000.0, 3340000.0, 3690000.0,
            -3954000.0, 3360000.0, 3710000.0
        );
        $results = $idx->search($q, 10);
        $ids = array_map(fn($r) => $r->getId(), $results);
        sort($ids);
        $this->assertEquals(["tokyo_skytree", "tokyo_tower"], $ids);
    }

    public function testGeo3dNearestQuery(): void
    {
        // k = 3 around Mt. Fuji returns Fuji + Tower + Skytree.
        $idx = $this->createGeo3dIndex();
        $q = Laurus\Geo3dNearestQuery::kNearest(
            "position", -3916073.0, 3437037.0, 3672751.0, 3
        );
        $results = $idx->search($q, 3);
        $this->assertCount(3, $results);
        $this->assertEquals("mt_fuji", $results[0]->getId());
        $ids = array_map(fn($r) => $r->getId(), $results);
        sort($ids);
        $this->assertEquals(["mt_fuji", "tokyo_skytree", "tokyo_tower"], $ids);
    }

    public function testGeo3dNearestQueryWithRadiusOptions(): void
    {
        // Verify the optional initial / max radius parameters are accepted.
        $idx = $this->createGeo3dIndex();
        $q = Laurus\Geo3dNearestQuery::kNearest(
            "position", -3955182.0, 3350553.0, 3700276.0, 2,
            10000.0,        // initial_radius_m
            10000000.0,     // max_radius_m
        );
        $results = $idx->search($q, 2);
        $ids = array_map(fn($r) => $r->getId(), $results);
        sort($ids);
        $this->assertEquals(["tokyo_skytree", "tokyo_tower"], $ids);
    }

    // ── Vector search ───────────────────────────────────────────────────

    public function testVectorQuery(): void
    {
        $idx = $this->createVectorIndex();
        $q = new Laurus\VectorQuery("embedding", [0.1, 0.2, 0.3, 0.4]);
        $results = $idx->search($q);
        $this->assertGreaterThanOrEqual(1, count($results));
        $this->assertEquals("doc1", $results[0]->getId());
    }

    // ── Hybrid search ───────────────────────────────────────────────────

    public function testSearchRequestLexicalOnly(): void
    {
        $idx = $this->createIndex();
        $req = new Laurus\SearchRequest(
            null, // query
            new Laurus\TermQuery("title", "rust"), // lexical_query
            null, // vector_query
            null, // filter_query
            null, // fusion
        );
        $results = $idx->search($req);
        $this->assertCount(1, $results);
    }

    // ── Fusion algorithms ───────────────────────────────────────────────

    public function testRRFRepr(): void
    {
        $rrf = new Laurus\RRF();
        $this->assertEquals("RRF(k=60)", (string)$rrf);
    }

    public function testWeightedSumRepr(): void
    {
        $ws = new Laurus\WeightedSum();
        $this->assertEquals("WeightedSum(lexical_weight=0.5, vector_weight=0.5)", (string)$ws);
    }

    // ── Analysis pipeline ───────────────────────────────────────────────

    public function testWhitespaceTokenizer(): void
    {
        $tok = new Laurus\WhitespaceTokenizer();
        $tokens = $tok->tokenize("hello world foo");
        $this->assertCount(3, $tokens);
        $this->assertEquals("hello", $tokens[0]->getText());
        $this->assertEquals("world", $tokens[1]->getText());
        $this->assertEquals("foo", $tokens[2]->getText());
    }

    public function testSynonymDictionary(): void
    {
        $dict = new Laurus\SynonymDictionary();
        $dict->addSynonymGroup(["quick", "fast", "speedy"]);
        $this->assertNotNull($dict);
    }

    public function testSynonymGraphFilter(): void
    {
        $dict = new Laurus\SynonymDictionary();
        $dict->addSynonymGroup(["happy", "joyful"]);
        $tok = new Laurus\WhitespaceTokenizer();
        $tokens = $tok->tokenize("I am happy");
        $filter = new Laurus\SynonymGraphFilter($dict);
        $expanded = $filter->apply($tokens);
        $texts = array_map(fn($t) => $t->getText(), $expanded);
        $this->assertContains("happy", $texts);
        $this->assertContains("joyful", $texts);
    }

    // ── SearchResult ────────────────────────────────────────────────────

    public function testSearchResultDocument(): void
    {
        $idx = $this->createIndex();
        $results = $idx->search("title:rust");
        $this->assertCount(1, $results);
        $doc = $results[0]->getDocument();
        $this->assertIsArray($doc);
        $this->assertEquals("Introduction to Rust", $doc["title"]);
    }

    // ── __toString ──────────────────────────────────────────────────────

    public function testTermQueryToString(): void
    {
        $q = new Laurus\TermQuery("title", "hello");
        $this->assertEquals("TermQuery(field='title', term='hello')", (string)$q);
    }

    public function testSearchResultToString(): void
    {
        $idx = $this->createIndex();
        $results = $idx->search("title:rust");
        $str = (string)$results[0];
        $this->assertStringContainsString("SearchResult(", $str);
        $this->assertStringContainsString("doc1", $str);
    }
}
