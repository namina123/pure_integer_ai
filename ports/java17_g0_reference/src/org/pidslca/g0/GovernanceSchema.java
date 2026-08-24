package org.pidslca.g0;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * G0b/G0c 的语言中立 schema reference。
 *
 * <p>本层只消费 GOV-CJSON-1 envelope bytes，冻结字段、标量、数组顺序与 genesis 规则；
 * 不执行 root pin、签名验证、权限判定、路径 I/O 或 capability 投影。</p>
 */
public final class GovernanceSchema {
    public static final String STATUS_REFERENCE_ONLY = GovCjson.STATUS_REFERENCE_ONLY;

    public static final int OK = 0;
    public static final int REJECT_EXACT_FIELDS = 101;
    public static final int REJECT_SCALAR = 102;

    public static final int SOURCE_REF_KEY_LENGTH = 11;
    public static final String ZERO_SHA256 = "0000000000000000000000000000000000000000000000000000000000000000";

    private static final Set<String> ROOT_REGISTRY_FIELDS = Set.of(
            "algorithm", "issuers", "key_id", "kind",
            "predecessor_registry_identity_sha256", "schema", "sequence", "version");
    private static final Set<String> ISSUER_RECORD_FIELDS = Set.of(
            "control_domain", "issuer_key_id", "not_after_registry_sequence",
            "not_before_registry_sequence", "public_key_hex", "role");
    private static final Set<String> REVOCATION_SNAPSHOT_FIELDS = Set.of(
            "algorithm", "key_id", "kind", "predecessor_revocation_identity_sha256",
            "registry_document_identity_sha256", "revocations", "schema", "sequence", "version");
    private static final Set<String> REVOCATION_RECORD_FIELDS = Set.of(
            "effective_declaration_sequence", "reason_digest_sha256", "revoked_key_id");
    private static final Set<String> SOURCE_SNAPSHOT_DECLARATION_FIELDS = Set.of(
            "algorithm", "control_domain", "ingest_code_identity_sha256", "key_id", "kind",
            "license_id", "license_review_artifact_identity_sha256", "metadata_byte_count",
            "metadata_id", "metadata_sha256", "official_uri",
            "predecessor_declaration_identity_sha256", "registry_document_identity_sha256",
            "revocation_document_identity_sha256", "schema", "sequence", "snapshot_id",
            "source_file_byte_count", "source_file_id", "source_file_sha256", "source_key",
            "source_ref_key", "transform_code_identity_sha256", "upstream_digest_algorithm",
            "upstream_digest_hex", "version");

    private GovernanceSchema() {
    }

    /** 已按专属 schema 回读的只读 raw-value 视图，仍完全未验签。 */
    public static final class SchemaDocument {
        private final String kind;
        private final String keyId;
        private final long sequence;
        private final GovCjson.Bytes canonicalSignedPayload;
        private final GovCjson.Bytes detachedSignature;
        private final GovCjson.Bytes domainPrefix;
        private final GovCjson.Bytes message;
        private final GovCjson.Bytes documentIdentity;

        private SchemaDocument(GovCjson.WireEnvelope envelope, long sequence) {
            this.kind = envelope.kind();
            this.keyId = envelope.keyId();
            this.sequence = sequence;
            this.canonicalSignedPayload = envelope.canonicalSignedPayload();
            this.detachedSignature = envelope.detachedSignature();
            this.domainPrefix = envelope.domainPrefix();
            this.message = envelope.message();
            this.documentIdentity = envelope.documentIdentity();
        }

        public String kind() {
            return kind;
        }

        public String keyId() {
            return keyId;
        }

        public long sequence() {
            return sequence;
        }

        public GovCjson.Bytes canonicalSignedPayload() {
            return canonicalSignedPayload;
        }

        public GovCjson.Bytes detachedSignature() {
            return detachedSignature;
        }

        public GovCjson.Bytes domainPrefix() {
            return domainPrefix;
        }

        public GovCjson.Bytes message() {
            return message;
        }

        public GovCjson.Bytes documentIdentity() {
            return documentIdentity;
        }

        public String status() {
            return STATUS_REFERENCE_ONLY;
        }
    }

    /** 解析任一当前 G0b/G0c schema document；不进行签名或链验证。 */
    public static SchemaDocument parseDocument(GovCjson.Bytes payload) {
        GovCjson.WireEnvelope envelope = GovCjson.parseEnvelope(payload);
        GovCjson.ObjectValue signedPayload = GovCjson.parse(envelope.canonicalSignedPayload());
        long sequence = validatePayload(signedPayload, envelope.kind());
        String keyId = requireKeyId(signedPayload.value("key_id"), "schema document key_id");
        if (!keyId.equals(envelope.keyId())) {
            failScalar("schema document key_id drift");
        }
        return new SchemaDocument(envelope, sequence);
    }

    /** 解析精确 root-registry schema，拒绝其他 kind 的跨域重放。 */
    public static SchemaDocument parseRootRegistry(GovCjson.Bytes payload) {
        SchemaDocument document = parseDocument(payload);
        if (!GovCjson.ROOT_REGISTRY.equals(document.kind())) {
            failExact("expected root-registry");
        }
        return document;
    }

    /** 解析精确 revocation-snapshot schema，仍不验证 registry 链。 */
    public static SchemaDocument parseRevocationSnapshot(GovCjson.Bytes payload) {
        SchemaDocument document = parseDocument(payload);
        if (!GovCjson.REVOCATION_SNAPSHOT.equals(document.kind())) {
            failExact("expected revocation-snapshot");
        }
        return document;
    }

    /** 解析精确 source declaration schema，绝不读取其指向的 source/metadata。 */
    public static SchemaDocument parseSourceSnapshotDeclaration(GovCjson.Bytes payload) {
        SchemaDocument document = parseDocument(payload);
        if (!GovCjson.SOURCE_SNAPSHOT_DECLARATION.equals(document.kind())) {
            failExact("expected source-snapshot-declaration");
        }
        return document;
    }

    private static long validatePayload(GovCjson.ObjectValue payload, String kind) {
        if (GovCjson.ROOT_REGISTRY.equals(kind)) {
            return validateRootRegistry(payload);
        }
        if (GovCjson.REVOCATION_SNAPSHOT.equals(kind)) {
            return validateRevocationSnapshot(payload);
        }
        if (GovCjson.SOURCE_SNAPSHOT_DECLARATION.equals(kind)) {
            return validateSourceSnapshotDeclaration(payload);
        }
        failExact("schema reference does not support this document kind");
        throw new AssertionError("unreachable");
    }

    private static long validateRootRegistry(GovCjson.ObjectValue payload) {
        requireExact(payload, ROOT_REGISTRY_FIELDS, "root-registry");
        Common common = requireCommon(payload, GovCjson.ROOT_REGISTRY);
        requirePredecessor(payload.value("predecessor_registry_identity_sha256"), common.sequence(),
                "root-registry predecessor");
        validateIssuers(payload.value("issuers"));
        return common.sequence();
    }

    private static long validateRevocationSnapshot(GovCjson.ObjectValue payload) {
        requireExact(payload, REVOCATION_SNAPSHOT_FIELDS, "revocation-snapshot");
        Common common = requireCommon(payload, GovCjson.REVOCATION_SNAPSHOT);
        requirePredecessor(payload.value("predecessor_revocation_identity_sha256"), common.sequence(),
                "revocation-snapshot predecessor");
        requireSha256(payload.value("registry_document_identity_sha256"),
                "revocation registry identity", false);
        validateRevocations(payload.value("revocations"));
        return common.sequence();
    }

    private static long validateSourceSnapshotDeclaration(GovCjson.ObjectValue payload) {
        requireExact(payload, SOURCE_SNAPSHOT_DECLARATION_FIELDS, "source-snapshot-declaration");
        Common common = requireCommon(payload, GovCjson.SOURCE_SNAPSHOT_DECLARATION);
        requireKeyId(payload.value("control_domain"), "declaration control_domain");
        for (String field : List.of(
                "ingest_code_identity_sha256",
                "license_review_artifact_identity_sha256",
                "metadata_sha256",
                "registry_document_identity_sha256",
                "revocation_document_identity_sha256",
                "source_file_sha256",
                "transform_code_identity_sha256")) {
            requireSha256(payload.value(field), "declaration " + field, false);
        }
        requirePredecessor(payload.value("predecessor_declaration_identity_sha256"), common.sequence(),
                "declaration predecessor");
        requireU63(payload.value("metadata_byte_count"), "metadata_byte_count", true);
        requireU63(payload.value("source_file_byte_count"), "source_file_byte_count", true);
        for (String field : List.of("metadata_id", "snapshot_id", "source_file_id", "source_key")) {
            requireOpaqueId(payload.value(field), "declaration " + field);
        }
        requireStrictHttpsLocator(payload.value("official_uri"), "declaration official_uri");
        requireLicenseId(payload.value("license_id"), "declaration license_id");
        validateSourceRefKey(payload.value("source_ref_key"));
        validateUpstreamDigest(payload);
        return common.sequence();
    }

    private static Common requireCommon(GovCjson.ObjectValue payload, String kind) {
        if (!equalsText(payload.value("algorithm"), GovCjson.ALGORITHM)
                || !equalsText(payload.value("kind"), kind)
                || requireU63(payload.value("schema"), "schema", false) != GovCjson.SCHEMA
                || requireU63(payload.value("version"), "version", false) != GovCjson.VERSION) {
            failScalar("signed_payload common constants drift");
        }
        return new Common(
                requireKeyId(payload.value("key_id"), "signed_payload key_id"),
                requireU63(payload.value("sequence"), "signed_payload sequence", true));
    }

    private static void validateIssuers(GovCjson.Value value) {
        List<GovCjson.Value> issuers = requireArray(value, "root-registry issuers");
        if (issuers.isEmpty()) {
            failScalar("root-registry issuers must be non-empty");
        }
        String previousKey = null;
        Set<String> publicKeys = new HashSet<>();
        for (int index = 0; index < issuers.size(); index++) {
            GovCjson.ObjectValue record = requireObject(issuers.get(index), "issuer record");
            requireExact(record, ISSUER_RECORD_FIELDS, "issuer record");
            String issuerKey = requireKeyId(record.value("issuer_key_id"), "issuer key_id");
            if (previousKey != null && GovCjson.compareAscii(issuerKey, previousKey) <= 0) {
                failScalar("issuer records are not strictly issuer_key_id ordered");
            }
            previousKey = issuerKey;
            String publicKey = requirePublicKey(record.value("public_key_hex"), "issuer public_key_hex");
            if (!publicKeys.add(publicKey)) {
                failScalar("issuer public_key_hex must be unique");
            }
            requireKeyId(record.value("control_domain"), "issuer control_domain");
            long before = requireU63(record.value("not_before_registry_sequence"),
                    "issuer not_before_registry_sequence", true);
            long after = requireU63(record.value("not_after_registry_sequence"),
                    "issuer not_after_registry_sequence", true);
            if (before > after) {
                failScalar("issuer validity window is inverted");
            }
            String role = requireText(record.value("role"), "issuer role");
            if (!"SOURCE_SNAPSHOT".equals(role) && !"ANNOTATION_SOURCE".equals(role)) {
                failScalar("issuer role is not registered");
            }
        }
    }

    private static void validateRevocations(GovCjson.Value value) {
        List<GovCjson.Value> revocations = requireArray(value, "revocations");
        String previousKey = null;
        for (int index = 0; index < revocations.size(); index++) {
            GovCjson.ObjectValue record = requireObject(revocations.get(index), "revocation record");
            requireExact(record, REVOCATION_RECORD_FIELDS, "revocation record");
            String revokedKey = requireKeyId(record.value("revoked_key_id"), "revoked_key_id");
            if (previousKey != null && GovCjson.compareAscii(revokedKey, previousKey) <= 0) {
                failScalar("revocations are not strictly revoked_key_id ordered");
            }
            previousKey = revokedKey;
            requireU63(record.value("effective_declaration_sequence"),
                    "effective_declaration_sequence", true);
            requireSha256(record.value("reason_digest_sha256"), "reason_digest_sha256", false);
        }
    }

    private static void validateSourceRefKey(GovCjson.Value value) {
        List<GovCjson.Value> fields = requireArray(value, "source_ref_key");
        if (fields.size() != SOURCE_REF_KEY_LENGTH) {
            failScalar("source_ref_key must contain exactly eleven u63 values");
        }
        long[] raw = new long[SOURCE_REF_KEY_LENGTH];
        for (int index = 0; index < raw.length; index++) {
            raw[index] = requireU63(fields.get(index), "source_ref_key[" + index + "]", false);
        }
        if (raw[0] == 0 || raw[1] == 0) {
            failScalar("source_ref_key source_kind/source_id must be positive");
        }
        if (raw[3] != 0 || raw[4] != 0 || raw[5] != 0 || raw[6] != 1) {
            failScalar("source_ref_key owner/visibility must be [0,0,0,1]");
        }
    }

    private static void validateUpstreamDigest(GovCjson.ObjectValue payload) {
        String algorithm = requireText(payload.value("upstream_digest_algorithm"), "upstream_digest_algorithm");
        String digest = requireText(payload.value("upstream_digest_hex"), "upstream_digest_hex");
        int expectedLength = switch (algorithm) {
            case "NONE" -> 0;
            case "SHA1" -> 40;
            case "SHA256" -> 64;
            default -> -1;
        };
        if (expectedLength < 0 || digest.length() != expectedLength || !isLowerHex(digest)) {
            failScalar("upstream digest does not match its algorithm");
        }
    }

    private static String requireKeyId(GovCjson.Value value, String label) {
        String text = requireText(value, label);
        if (text.length() < 1 || text.length() > 64 || !isAsciiLower(text.charAt(0))) {
            failScalar(label + " is not a key_id/control_domain");
        }
        for (int index = 0; index < text.length(); index++) {
            char character = text.charAt(index);
            if (!(isAsciiLower(character) || isAsciiDigit(character)
                    || character == '.' || character == '_' || character == ':' || character == '-')) {
                failScalar(label + " is not a key_id/control_domain");
            }
        }
        return text;
    }

    private static String requireOpaqueId(GovCjson.Value value, String label) {
        String text = requireText(value, label);
        if (text.length() < 1 || text.length() > 128
                || !(isAsciiLower(text.charAt(0)) || isAsciiUpper(text.charAt(0)))) {
            failScalar(label + " is not an opaque_id");
        }
        for (int index = 0; index < text.length(); index++) {
            char character = text.charAt(index);
            if (!(isAsciiLower(character) || isAsciiUpper(character) || isAsciiDigit(character)
                    || character == '.' || character == '_' || character == ':' || character == '-')) {
                failScalar(label + " is not an opaque_id");
            }
        }
        return text;
    }

    private static String requireLicenseId(GovCjson.Value value, String label) {
        String text = requireText(value, label);
        if (text.length() < 1 || text.length() > 128) {
            failScalar(label + " is not a license_id");
        }
        for (int index = 0; index < text.length(); index++) {
            char character = text.charAt(index);
            if (!(isAsciiLower(character) || isAsciiUpper(character) || isAsciiDigit(character)
                    || character == '.' || character == '-')) {
                failScalar(label + " is not a license_id");
            }
        }
        return text;
    }

    private static String requireSha256(GovCjson.Value value, String label, boolean allowZero) {
        String text = requireText(value, label);
        if (text.length() != 64 || !isLowerHex(text) || (!allowZero && ZERO_SHA256.equals(text))) {
            failScalar(label + " is not an allowed SHA-256");
        }
        return text;
    }

    private static String requirePredecessor(GovCjson.Value value, long sequence, String label) {
        String digest = requireSha256(value, label, true);
        if ((sequence == 1 && !ZERO_SHA256.equals(digest))
                || (sequence != 1 && ZERO_SHA256.equals(digest))) {
            failScalar(label + " violates the genesis rule");
        }
        return digest;
    }

    private static String requirePublicKey(GovCjson.Value value, String label) {
        String text = requireText(value, label);
        if (text.length() != 64 || !isLowerHex(text)) {
            failScalar(label + " is not a 32-byte lowercase public key");
        }
        return text;
    }

    private static String requireStrictHttpsLocator(GovCjson.Value value, String label) {
        String text = requireText(value, label);
        if (!text.startsWith("https://")) {
            failScalar(label + " must begin with https://");
        }
        String authorityAndTail = text.substring("https://".length());
        int slash = authorityAndTail.indexOf('/');
        String host = slash < 0 ? authorityAndTail : authorityAndTail.substring(0, slash);
        String tail = slash < 0 ? "" : authorityAndTail.substring(slash);
        if (host.isEmpty() || host.length() > 253) {
            failScalar(label + " host length is invalid");
        }
        String[] labels = host.split("\\.", -1);
        for (String item : labels) {
            if (item.isEmpty() || item.length() > 63
                    || !(isAsciiLower(item.charAt(0)) || isAsciiDigit(item.charAt(0)))
                    || !(isAsciiLower(item.charAt(item.length() - 1)) || isAsciiDigit(item.charAt(item.length() - 1)))) {
                failScalar(label + " host is not a lowercase ASCII DNS A-label");
            }
            for (int index = 0; index < item.length(); index++) {
                char character = item.charAt(index);
                if (!(isAsciiLower(character) || isAsciiDigit(character) || character == '-')) {
                    failScalar(label + " host is not a lowercase ASCII DNS A-label");
                }
            }
        }
        for (int index = 0; index < tail.length();) {
            char character = tail.charAt(index);
            if (!isUriTailCharacter(character)) {
                failScalar(label + " path/query/fragment contains an invalid character");
            }
            if (character == '%') {
                if (index + 2 >= tail.length() || !isAsciiUpperHex(tail.charAt(index + 1))
                        || !isAsciiUpperHex(tail.charAt(index + 2))) {
                    failScalar(label + " percent-encoding is not uppercase hex");
                }
                index += 3;
            } else {
                index++;
            }
        }
        return text;
    }

    private static GovCjson.ObjectValue requireObject(GovCjson.Value value, String label) {
        if (!(value instanceof GovCjson.ObjectValue object)) {
            failScalar(label + " must be an object");
            throw new AssertionError("unreachable");
        }
        return object;
    }

    private static List<GovCjson.Value> requireArray(GovCjson.Value value, String label) {
        if (!(value instanceof GovCjson.ArrayValue array)) {
            failScalar(label + " must be an array");
            throw new AssertionError("unreachable");
        }
        return array.values();
    }

    private static String requireText(GovCjson.Value value, String label) {
        return GovCjson.requireText(value, label, REJECT_SCALAR);
    }

    private static long requireU63(GovCjson.Value value, String label, boolean positive) {
        long result = GovCjson.requireU63(value, label, REJECT_SCALAR);
        if (positive && result == 0) {
            failScalar(label + " must be positive");
        }
        return result;
    }

    private static void requireExact(GovCjson.ObjectValue value, Set<String> fields, String label) {
        if (!value.values().keySet().equals(fields)) {
            failExact(label + " field set is not exact");
        }
    }

    private static boolean equalsText(GovCjson.Value value, String expected) {
        return value instanceof GovCjson.TextValue text && expected.equals(text.value());
    }

    private static boolean isAsciiLower(char character) {
        return character >= 'a' && character <= 'z';
    }

    private static boolean isAsciiUpper(char character) {
        return character >= 'A' && character <= 'Z';
    }

    private static boolean isAsciiDigit(char character) {
        return character >= '0' && character <= '9';
    }

    private static boolean isAsciiUpperHex(char character) {
        return isAsciiDigit(character) || (character >= 'A' && character <= 'F');
    }

    private static boolean isLowerHex(String value) {
        for (int index = 0; index < value.length(); index++) {
            char character = value.charAt(index);
            if (!(isAsciiDigit(character) || (character >= 'a' && character <= 'f'))) {
                return false;
            }
        }
        return true;
    }

    private static boolean isUriTailCharacter(char character) {
        return isAsciiLower(character) || isAsciiUpper(character) || isAsciiDigit(character)
                || "-._~!$&'()*+,;=:@/?#%".indexOf(character) >= 0;
    }

    private static void failExact(String message) {
        throw new GovCjson.GovException(REJECT_EXACT_FIELDS, message);
    }

    private static void failScalar(String message) {
        throw new GovCjson.GovException(REJECT_SCALAR, message);
    }

    private record Common(String keyId, long sequence) {
    }
}
