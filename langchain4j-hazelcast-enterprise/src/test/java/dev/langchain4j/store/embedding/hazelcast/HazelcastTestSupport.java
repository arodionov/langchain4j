package dev.langchain4j.store.embedding.hazelcast;

/**
 * Shared helpers for Hazelcast Enterprise integration tests.
 */
final class HazelcastTestSupport {

    private HazelcastTestSupport() {}

    /**
     * Resolves the Hazelcast Enterprise license key from the
     * {@code hazelcast.enterprise.license.key} system property, falling back to the
     * {@code HZ_LICENSEKEY} environment variable.
     *
     * @return the license key, or {@code null} if neither source is set
     */
    static String licenseKey() {
        String key = System.getProperty("hazelcast.enterprise.license.key");
        if (key == null || key.isBlank()) {
            key = System.getenv("HZ_LICENSEKEY");
        }
        return key;
    }
}
