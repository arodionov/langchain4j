package dev.langchain4j.store.embedding.hazelcast;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.hazelcast.config.Config;
import com.hazelcast.config.vector.Metric;
import com.hazelcast.config.vector.VectorCollectionConfig;
import com.hazelcast.config.vector.VectorIndexConfig;
import com.hazelcast.config.JoinConfig;
import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.vector.VectorCollection;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;

/**
 * Integration tests for {@link HazelcastEmbeddingStore}.
 * <p>
 * Requires a running Hazelcast Enterprise instance with a valid license key configured
 * as the system property {@code hazelcast.enterprise.license.key} or via the
 * {@code HZ_LICENSEKEY} environment variable.
 * <p>
 * An embedded single-member cluster is used — no external process needed.
 */
class HazelcastEmbeddingStoreIT {

    /**
     * Small fixed dimension used throughout tests.
     * Must match the dimension of the test embeddings below.
     */
    private static final int DIMENSION = 4;

    static HazelcastInstance hazelcastInstance;

    HazelcastEmbeddingStore store;

    @BeforeAll
    static void startHazelcast() {
        String licenseKey = System.getProperty("hazelcast.enterprise.license.key");
        if (licenseKey == null || licenseKey.isBlank()) {
            licenseKey = System.getenv("HZ_LICENSEKEY");
        }
        assumeTrue(
                licenseKey != null && !licenseKey.isBlank(),
                "Hazelcast Enterprise license key not provided via the "
                        + "'hazelcast.enterprise.license.key' system property or the 'HZ_LICENSEKEY' "
                        + "environment variable; skipping HazelcastEmbeddingStoreIT.");

        Config config = new Config();
        config.setClusterName("langchain4j-embedding-test");
        config.setLicenseKey(licenseKey);
        JoinConfig join = config.getNetworkConfig().getJoin();
        join.getMulticastConfig().setEnabled(false);
        join.getTcpIpConfig().setEnabled(false);
        hazelcastInstance = Hazelcast.newHazelcastInstance(config);
    }

    @AfterAll
    static void stopHazelcast() {
        if (hazelcastInstance != null) {
            hazelcastInstance.shutdown();
        }
    }

    @BeforeEach
    void setUp(TestInfo testInfo) {
        // Use the test display name as the collection name for isolation
        store = HazelcastEmbeddingStore.builder()
                .hazelcastInstance(hazelcastInstance)
                .collectionName(testInfo.getDisplayName())
                .dimension(DIMENSION)
                .metric(Metric.COSINE)
                .build();
    }

    @AfterEach
    void tearDown() {
        if (store != null) {
            store.collection.destroy();
        }
    }

    // -------------------------------------------------------------------------
    // add / search
    // -------------------------------------------------------------------------

    @Test
    void should_add_embedding_and_return_generated_id() {
        Embedding embedding = embedding(1f, 0f, 0f, 0f);
        String id = store.add(embedding);
        assertThat(id).isNotBlank();
    }

    @Test
    void should_add_embedding_with_explicit_id() {
        Embedding embedding = embedding(1f, 0f, 0f, 0f);
        store.add("my-id", embedding);

        EmbeddingSearchResult<TextSegment> result = store.search(searchRequest(embedding, 1));
        assertThat(result.matches()).hasSize(1);
        assertThat(result.matches().get(0).embeddingId()).isEqualTo("my-id");
    }

    @Test
    void should_add_embedding_with_segment_and_retrieve_it() {
        TextSegment segment = TextSegment.from("Hello, world!");
        Embedding embedding = embedding(1f, 0f, 0f, 0f);

        store.add(embedding, segment);

        EmbeddingSearchResult<TextSegment> result = store.search(searchRequest(embedding, 1));
        assertThat(result.matches()).hasSize(1);
        assertThat(result.matches().get(0).embedded()).isEqualTo(segment);
    }

    @Test
    void should_add_all_embeddings_without_segments() {
        List<Embedding> embeddings = List.of(
                embedding(1f, 0f, 0f, 0f),
                embedding(0f, 1f, 0f, 0f));

        List<String> ids = store.addAll(embeddings);
        assertThat(ids).hasSize(2);
    }

    @Test
    void should_add_all_embeddings_with_segments() {
        List<Embedding> embeddings = List.of(
                embedding(1f, 0f, 0f, 0f),
                embedding(0f, 1f, 0f, 0f));
        List<TextSegment> segments = List.of(
                TextSegment.from("First"),
                TextSegment.from("Second"));

        List<String> ids = store.addAll(embeddings, segments);
        assertThat(ids).hasSize(2);
    }

    @Test
    void should_return_matches_ordered_by_descending_score() {
        Embedding query    = embedding(1f, 0f, 0f, 0f);
        Embedding close    = embedding(0.9f, 0.1f, 0f, 0f);
        Embedding far      = embedding(0f, 0f, 1f, 0f);

        store.add(close, TextSegment.from("close"));
        store.add(far,   TextSegment.from("far"));

        EmbeddingSearchResult<TextSegment> result = store.search(searchRequest(query, 2));
        assertThat(result.matches()).hasSize(2);
        assertThat(result.matches().get(0).score())
                .isGreaterThan(result.matches().get(1).score());
        assertThat(result.matches().get(0).embedded().text()).isEqualTo("close");
    }

    @Test
    void should_respect_max_results() {
        store.add(embedding(1f, 0f, 0f, 0f), TextSegment.from("A"));
        store.add(embedding(0f, 1f, 0f, 0f), TextSegment.from("B"));
        store.add(embedding(0f, 0f, 1f, 0f), TextSegment.from("C"));

        EmbeddingSearchResult<TextSegment> result =
                store.search(searchRequest(embedding(1f, 0f, 0f, 0f), 2));
        assertThat(result.matches()).hasSize(2);
    }

    @Test
    void should_respect_min_score() {
        store.add(embedding(1f, 0f, 0f, 0f), TextSegment.from("exact"));
        store.add(embedding(0f, 1f, 0f, 0f), TextSegment.from("orthogonal"));

        EmbeddingSearchResult<TextSegment> result = store.search(
                EmbeddingSearchRequest.builder()
                        .queryEmbedding(embedding(1f, 0f, 0f, 0f))
                        .maxResults(10)
                        .minScore(0.9)
                        .build());

        assertThat(result.matches()).hasSize(1);
        assertThat(result.matches().get(0).embedded().text()).isEqualTo("exact");
    }

    @Test
    void should_report_relevance_score_in_normalized_range() {
        // An orthogonal vector has cosine similarity 0. Hazelcast normalizes COSINE scores
        // to a non-negative [0, 1] range, so the relevance score must be ~0.5 — NOT ~0.75,
        // which is what double-normalizing through RelevanceScore.fromCosineSimilarity produces.
        // This pins the score semantics that the order-only / threshold tests cannot detect.
        Embedding query = embedding(1f, 0f, 0f, 0f);
        Embedding orthogonal = embedding(0f, 1f, 0f, 0f);

        store.add(orthogonal, TextSegment.from("orthogonal"));

        EmbeddingSearchResult<TextSegment> result = store.search(searchRequest(query, 1));
        assertThat(result.matches()).hasSize(1);
        assertThat(result.matches().get(0).score()).isBetween(0.45, 0.55);
    }

    // -------------------------------------------------------------------------
    // remove
    // -------------------------------------------------------------------------

    @Test
    void should_remove_by_id() {
        String id = store.add(embedding(1f, 0f, 0f, 0f), TextSegment.from("to remove"));
        store.remove(id);

        EmbeddingSearchResult<TextSegment> result =
                store.search(searchRequest(embedding(1f, 0f, 0f, 0f), 10));
        assertThat(result.matches())
                .noneMatch(m -> id.equals(m.embeddingId()));
    }

    @Test
    void should_remove_all_by_ids() {
        String id1 = store.add(embedding(1f, 0f, 0f, 0f), TextSegment.from("one"));
        String id2 = store.add(embedding(0f, 1f, 0f, 0f), TextSegment.from("two"));
        store.add(embedding(0f, 0f, 1f, 0f), TextSegment.from("three"));

        store.removeAll(List.of(id1, id2));

        EmbeddingSearchResult<TextSegment> result =
                store.search(searchRequest(embedding(0f, 0f, 1f, 0f), 10));
        assertThat(result.matches())
                .noneMatch(m -> id1.equals(m.embeddingId()))
                .noneMatch(m -> id2.equals(m.embeddingId()));
    }

    @Test
    void should_remove_all_and_remain_usable() {
        store.add(embedding(1f, 0f, 0f, 0f), TextSegment.from("one"));
        store.add(embedding(0f, 1f, 0f, 0f), TextSegment.from("two"));

        // when all entries are cleared
        store.removeAll();

        // then the collection is empty...
        EmbeddingSearchResult<TextSegment> afterClear =
                store.search(searchRequest(embedding(1f, 0f, 0f, 0f), 10));
        assertThat(afterClear.matches()).isEmpty();

        // ...and the store is still usable (clearAsync preserves the collection and its index)
        String id = store.add(embedding(1f, 0f, 0f, 0f), TextSegment.from("after clear"));
        EmbeddingSearchResult<TextSegment> afterReuse =
                store.search(searchRequest(embedding(1f, 0f, 0f, 0f), 10));
        assertThat(afterReuse.matches()).hasSize(1);
        assertThat(afterReuse.matches().get(0).embeddingId()).isEqualTo(id);
        assertThat(afterReuse.matches().get(0).embedded().text()).isEqualTo("after clear");
    }

    @Test
    void removeAll_filter_throws_unsupported() {
        assertThatThrownBy(() -> store.removeAll((dev.langchain4j.store.embedding.filter.Filter) null))
                .isInstanceOf(UnsupportedFeatureException.class);
    }

    // -------------------------------------------------------------------------
    // Builder validation
    // -------------------------------------------------------------------------

    @Test
    void builder_throws_when_hazelcast_instance_missing() {
        assertThatThrownBy(() -> HazelcastEmbeddingStore.builder()
                .collectionName("test")
                .dimension(4)
                .build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("hazelcastInstance");
    }

    @Test
    void builder_throws_when_dimension_not_set() {
        assertThatThrownBy(() -> HazelcastEmbeddingStore.builder()
                .hazelcastInstance(hazelcastInstance)
                .collectionName("test")
                .build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("dimension");
    }

    @Test
    void should_accept_pre_configured_collection() {
        VectorCollectionConfig config = new VectorCollectionConfig("pre-configured")
                .addVectorIndexConfig(new VectorIndexConfig()
                        .setDimension(DIMENSION)
                        .setMetric(Metric.COSINE));
        VectorCollection<String, TextSegmentDocument> col =
                VectorCollection.getCollection(hazelcastInstance, config);

        HazelcastEmbeddingStore s = HazelcastEmbeddingStore.create(col);
        String id = s.add(embedding(1f, 0f, 0f, 0f), TextSegment.from("pre-configured"));
        assertThat(id).isNotBlank();

        col.destroy();
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static Embedding embedding(float... values) {
        return new Embedding(values);
    }

    private static EmbeddingSearchRequest searchRequest(Embedding query, int maxResults) {
        return EmbeddingSearchRequest.builder()
                .queryEmbedding(query)
                .maxResults(maxResults)
                .build();
    }
}
