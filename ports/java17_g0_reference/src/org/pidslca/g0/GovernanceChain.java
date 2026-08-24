package org.pidslca.g0;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * G0b-1 的纯 chain-shape reference。
 *
 * <p>它只验证完整 canonical envelope collection 的 sequence、predecessor、scope 及累计
 * revocation 结构。它不验证签名、root pin、issuer role/domain/window、撤销 cutoff，也绝不
 * 接收 caller 提供的 VALID verdict 或输出 capability。</p>
 */
public final class GovernanceChain {
    public static final String STATUS_REFERENCE_ONLY = GovernanceSchema.STATUS_REFERENCE_ONLY;

    public static final int MAX_DOCUMENTS_TOTAL = 1_024;
    public static final int MAX_INPUT_BYTES = 8_388_608;

    public static final int OK = 0;
    public static final int REJECT_REGISTRY_CHAIN = 105;
    public static final int REJECT_REVOCATION_REGISTRY_BINDING = 108;
    public static final int REJECT_REVOCATION_CHAIN = 109;
    public static final int REJECT_REVOCATION_SET_OR_EFFECTIVE_SEQUENCE = 110;
    public static final int REJECT_DECLARATION_REGISTRY_BINDING = 111;
    public static final int REJECT_DECLARATION_REVOCATION_BINDING = 112;
    public static final int REJECT_DECLARATION_CHAIN = 114;
    public static final int REJECT_INPUT_COLLECTION = 117;

    private GovernanceChain() {
    }

    /** 成功只暴露三条链的 derived head identity，不带 payload 或 trust/capability 含义。 */
    public record ChainHeads(
            GovCjson.Bytes registryHeadIdentity,
            GovCjson.Bytes revocationHeadIdentity,
            GovCjson.Bytes declarationHeadIdentity) {
        public String status() {
            return STATUS_REFERENCE_ONLY;
        }
    }

    /**
     * 验证三组无序、有限 canonical envelope bytes。
     *
     * <p>collection 的 Java {@link List} 仅是宿主传参方式；链一律由 signed sequence、
     * predecessor 和 derived identity 重建。</p>
     */
    public static ChainHeads validate(
            List<GovCjson.Bytes> registryDocuments,
            List<GovCjson.Bytes> revocationDocuments,
            List<GovCjson.Bytes> declarationDocuments) {
        requireInputCollections(registryDocuments, revocationDocuments, declarationDocuments);

        List<GovernanceSchema.SchemaDocument> registries = parseRegistries(sortedRawWitness(registryDocuments));
        List<GovernanceSchema.SchemaDocument> revocations = parseRevocations(sortedRawWitness(revocationDocuments));
        List<GovernanceSchema.SchemaDocument> declarations = parseDeclarations(sortedRawWitness(declarationDocuments));
        rejectDuplicateIdentities(registries, revocations, declarations);

        List<GovernanceSchema.SchemaDocument> orderedRegistries = validateRegistryChain(registries);
        List<GovernanceSchema.SchemaDocument> orderedRevocations = validateRevocationChain(
                revocations, orderedRegistries.get(orderedRegistries.size() - 1));
        List<GovernanceSchema.SchemaDocument> orderedDeclarations = validateDeclarationChain(
                declarations,
                orderedRegistries.get(orderedRegistries.size() - 1),
                orderedRevocations);
        return new ChainHeads(
                orderedRegistries.get(orderedRegistries.size() - 1).documentIdentity(),
                orderedRevocations.get(orderedRevocations.size() - 1).documentIdentity(),
                orderedDeclarations.get(orderedDeclarations.size() - 1).documentIdentity());
    }

    private static void requireInputCollections(
            List<GovCjson.Bytes> registryDocuments,
            List<GovCjson.Bytes> revocationDocuments,
            List<GovCjson.Bytes> declarationDocuments) {
        if (registryDocuments == null || revocationDocuments == null || declarationDocuments == null) {
            fail(REJECT_INPUT_COLLECTION, "chain collections must be non-null");
        }
        List<List<GovCjson.Bytes>> collections = List.of(
                registryDocuments, revocationDocuments, declarationDocuments);
        int totalDocuments = 0;
        long totalBytes = 0;
        for (List<GovCjson.Bytes> collection : collections) {
            if (collection.isEmpty()) {
                fail(REJECT_INPUT_COLLECTION, "chain collections must be non-empty");
            }
            if (collection.size() > MAX_DOCUMENTS_TOTAL - totalDocuments) {
                fail(REJECT_INPUT_COLLECTION, "chain document count exceeds the fixed budget");
            }
            totalDocuments += collection.size();
            for (GovCjson.Bytes document : collection) {
                if (document == null || document.length() == 0) {
                    fail(REJECT_INPUT_COLLECTION, "chain collection contains a non-empty envelope other than bytes");
                }
                totalBytes += document.length();
                if (totalBytes > MAX_INPUT_BYTES) {
                    fail(REJECT_INPUT_COLLECTION, "chain input byte count exceeds the fixed budget");
                }
            }
        }
    }

    private static List<GovernanceSchema.SchemaDocument> parseRegistries(List<GovCjson.Bytes> documents) {
        List<GovernanceSchema.SchemaDocument> result = new ArrayList<>();
        for (GovCjson.Bytes document : documents) {
            result.add(GovernanceSchema.parseRootRegistry(document));
        }
        return result;
    }

    /** 固定 raw envelope witness 顺序，避免 caller List 排列改变多失败输入的首个结果。 */
    private static List<GovCjson.Bytes> sortedRawWitness(List<GovCjson.Bytes> documents) {
        List<GovCjson.Bytes> result = new ArrayList<>(documents);
        result.sort(GovCjson::compareUnsignedBytes);
        return result;
    }

    private static List<GovernanceSchema.SchemaDocument> parseRevocations(List<GovCjson.Bytes> documents) {
        List<GovernanceSchema.SchemaDocument> result = new ArrayList<>();
        for (GovCjson.Bytes document : documents) {
            result.add(GovernanceSchema.parseRevocationSnapshot(document));
        }
        return result;
    }

    private static List<GovernanceSchema.SchemaDocument> parseDeclarations(List<GovCjson.Bytes> documents) {
        List<GovernanceSchema.SchemaDocument> result = new ArrayList<>();
        for (GovCjson.Bytes document : documents) {
            result.add(GovernanceSchema.parseSourceSnapshotDeclaration(document));
        }
        return result;
    }

    private static void rejectDuplicateIdentities(
            List<GovernanceSchema.SchemaDocument> registries,
            List<GovernanceSchema.SchemaDocument> revocations,
            List<GovernanceSchema.SchemaDocument> declarations) {
        Set<GovCjson.Bytes> seen = new HashSet<>();
        for (List<GovernanceSchema.SchemaDocument> collection : List.of(
                registries, revocations, declarations)) {
            for (GovernanceSchema.SchemaDocument document : collection) {
                if (!seen.add(document.documentIdentity())) {
                    fail(REJECT_INPUT_COLLECTION, "chain collection contains a duplicate derived identity");
                }
            }
        }
    }

    private static List<GovernanceSchema.SchemaDocument> validateRegistryChain(
            List<GovernanceSchema.SchemaDocument> documents) {
        List<GovernanceSchema.SchemaDocument> ordered = orderCompleteChain(
                documents,
                "predecessor_registry_identity_sha256",
                REJECT_REGISTRY_CHAIN,
                "root-registry");
        String rootKeyId = ordered.get(0).keyId();
        for (int index = 1; index < ordered.size(); index++) {
            if (!rootKeyId.equals(ordered.get(index).keyId())) {
                fail(REJECT_REGISTRY_CHAIN, "root-registry structural key_id scope drift");
            }
        }
        return ordered;
    }

    private static List<GovernanceSchema.SchemaDocument> validateRevocationChain(
            List<GovernanceSchema.SchemaDocument> documents,
            GovernanceSchema.SchemaDocument registryHead) {
        String expectedRegistryIdentity = registryHead.documentIdentity().toLowerHex();
        for (GovernanceSchema.SchemaDocument document : documents) {
            GovCjson.ObjectValue payload = signedPayload(document);
            if (!expectedRegistryIdentity.equals(text(payload.value("registry_document_identity_sha256")))
                    || !registryHead.keyId().equals(document.keyId())) {
                fail(REJECT_REVOCATION_REGISTRY_BINDING,
                        "revocation-snapshot is not bound to the registry structural scope");
            }
        }

        List<GovernanceSchema.SchemaDocument> ordered = orderCompleteChain(
                documents,
                "predecessor_revocation_identity_sha256",
                REJECT_REVOCATION_CHAIN,
                "revocation-snapshot");
        Set<String> allowedIssuerKeys = issuerKeyIds(registryHead);
        Map<String, RevocationRecord> previousRecords = Map.of();
        for (GovernanceSchema.SchemaDocument document : ordered) {
            Map<String, RevocationRecord> currentRecords = new HashMap<>();
            GovCjson.ArrayValue revocations = array(signedPayload(document).value("revocations"));
            for (GovCjson.Value rawRecord : revocations.values()) {
                GovCjson.ObjectValue record = object(rawRecord);
                String revokedKeyId = text(record.value("revoked_key_id"));
                if (!allowedIssuerKeys.contains(revokedKeyId)) {
                    fail(REJECT_REVOCATION_SET_OR_EFFECTIVE_SEQUENCE,
                            "revocation record references an issuer absent from the bound registry");
                }
                currentRecords.put(revokedKeyId, new RevocationRecord(
                        unsigned(record.value("effective_declaration_sequence")),
                        text(record.value("reason_digest_sha256"))));
            }
            for (Map.Entry<String, RevocationRecord> entry : previousRecords.entrySet()) {
                if (!Objects.equals(currentRecords.get(entry.getKey()), entry.getValue())) {
                    fail(REJECT_REVOCATION_SET_OR_EFFECTIVE_SEQUENCE,
                            "revocation chain deletes or rewrites a cumulative record");
                }
            }
            if (document.sequence() > 1 && currentRecords.size() <= previousRecords.size()) {
                fail(REJECT_REVOCATION_SET_OR_EFFECTIVE_SEQUENCE,
                        "revocation successor must add a cumulative record");
            }
            previousRecords = Map.copyOf(currentRecords);
        }
        return ordered;
    }

    private static List<GovernanceSchema.SchemaDocument> validateDeclarationChain(
            List<GovernanceSchema.SchemaDocument> documents,
            GovernanceSchema.SchemaDocument registryHead,
            List<GovernanceSchema.SchemaDocument> revocations) {
        String expectedRegistryIdentity = registryHead.documentIdentity().toLowerHex();
        for (GovernanceSchema.SchemaDocument document : documents) {
            if (!expectedRegistryIdentity.equals(text(signedPayload(document).value(
                    "registry_document_identity_sha256")))) {
                fail(REJECT_DECLARATION_REGISTRY_BINDING,
                        "source declaration is not bound to the registry head identity");
            }
        }

        List<GovernanceSchema.SchemaDocument> ordered = orderCompleteChain(
                documents,
                "predecessor_declaration_identity_sha256",
                REJECT_DECLARATION_CHAIN,
                "source-snapshot-declaration");
        String keyId = ordered.get(0).keyId();
        String kind = ordered.get(0).kind();
        for (int index = 1; index < ordered.size(); index++) {
            GovernanceSchema.SchemaDocument document = ordered.get(index);
            if (!keyId.equals(document.keyId()) || !kind.equals(document.kind())) {
                fail(REJECT_DECLARATION_CHAIN, "source declaration key_id/kind scope drift");
            }
        }

        Set<String> revocationIdentities = new HashSet<>();
        for (GovernanceSchema.SchemaDocument document : revocations) {
            revocationIdentities.add(document.documentIdentity().toLowerHex());
        }
        for (GovernanceSchema.SchemaDocument document : ordered) {
            if (!revocationIdentities.contains(text(signedPayload(document).value(
                    "revocation_document_identity_sha256")))) {
                fail(REJECT_DECLARATION_REVOCATION_BINDING,
                        "source declaration binds a revocation identity outside this collection");
            }
        }
        String selectedRevocationIdentity = revocations.get(revocations.size() - 1)
                .documentIdentity().toLowerHex();
        if (!selectedRevocationIdentity.equals(text(signedPayload(ordered.get(ordered.size() - 1)).value(
                "revocation_document_identity_sha256")))) {
            fail(REJECT_DECLARATION_REVOCATION_BINDING,
                    "source declaration head is not bound to the selected revocation head");
        }
        return ordered;
    }

    private static List<GovernanceSchema.SchemaDocument> orderCompleteChain(
            List<GovernanceSchema.SchemaDocument> documents,
            String predecessorField,
            int code,
            String label) {
        Map<Long, GovernanceSchema.SchemaDocument> bySequence = new HashMap<>();
        for (GovernanceSchema.SchemaDocument document : documents) {
            if (bySequence.putIfAbsent(document.sequence(), document) != null) {
                fail(code, label + " chain contains a duplicate sequence");
            }
        }
        List<GovernanceSchema.SchemaDocument> ordered = new ArrayList<>();
        for (long sequence = 1; sequence <= documents.size(); sequence++) {
            GovernanceSchema.SchemaDocument document = bySequence.get(sequence);
            if (document == null) {
                fail(code, label + " chain sequence is not complete");
            }
            if (!ordered.isEmpty()) {
                String predecessor = text(signedPayload(document).value(predecessorField));
                String expected = ordered.get(ordered.size() - 1).documentIdentity().toLowerHex();
                if (!expected.equals(predecessor)) {
                    fail(code, label + " chain predecessor identity is discontinuous");
                }
            }
            ordered.add(document);
        }
        return ordered;
    }

    private static Set<String> issuerKeyIds(GovernanceSchema.SchemaDocument registryHead) {
        GovCjson.ArrayValue issuers = array(signedPayload(registryHead).value("issuers"));
        Set<String> result = new HashSet<>();
        for (GovCjson.Value rawIssuer : issuers.values()) {
            result.add(text(object(rawIssuer).value("issuer_key_id")));
        }
        return result;
    }

    private static GovCjson.ObjectValue signedPayload(GovernanceSchema.SchemaDocument document) {
        return GovCjson.parse(document.canonicalSignedPayload());
    }

    private static GovCjson.ObjectValue object(GovCjson.Value value) {
        if (value instanceof GovCjson.ObjectValue result) {
            return result;
        }
        throw new IllegalStateException("schema-validated object unexpectedly changed shape");
    }

    private static GovCjson.ArrayValue array(GovCjson.Value value) {
        if (value instanceof GovCjson.ArrayValue result) {
            return result;
        }
        throw new IllegalStateException("schema-validated array unexpectedly changed shape");
    }

    private static String text(GovCjson.Value value) {
        if (value instanceof GovCjson.TextValue result) {
            return result.value();
        }
        throw new IllegalStateException("schema-validated text unexpectedly changed shape");
    }

    private static long unsigned(GovCjson.Value value) {
        if (value instanceof GovCjson.UIntValue result && result.value() >= 0) {
            return result.value();
        }
        throw new IllegalStateException("schema-validated u63 unexpectedly changed shape");
    }

    private static void fail(int code, String message) {
        throw new GovCjson.GovException(code, message);
    }

    private record RevocationRecord(long effectiveSequence, String reasonDigest) {
    }
}
