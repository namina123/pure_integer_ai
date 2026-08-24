package org.pidslca.g0;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 三份公开 G0 conformance fixture 的无依赖 Java 17 runner。
 *
 * <p>文件读取、UTF-8 与 fixture JSON parser 都只存在于本测试适配层。GOV-CJSON-1 core 不
 * 接收 {@link Path}，不执行 I/O，也不引用这个类。当前 runner 有意不实现或调用 Ed25519：
 * RFC 向量只核对公开 transport 长度与预期 verdict 的冻结形状。</p>
 */
public final class FixtureRunner {
    private static final String WIRE_FIXTURE = "gov_cjson_v1_conformance.json";
    private static final String SCHEMA_FIXTURE = "gov_g0b_g0c_schema_v1_conformance.json";
    private static final String CHAIN_FIXTURE = "gov_g0b_chain_shape_v1_conformance.json";
    private static final GovCjson.Bytes ZERO_SIGNATURE = GovCjson.Bytes.copyOf(new byte[64]);

    private FixtureRunner() {
    }

    public static void main(String[] args) {
        if (args.length != 1) {
            System.err.println("usage: FixtureRunner <public-fixture-directory>");
            System.exit(2);
        }
        try {
            PublicFixtureSource fixtureSource = new PublicFixtureSource(Path.of(args[0]));
            RunCounts counts = new RunCounts();
            ManifestCatalog catalog = ManifestCatalog.load(fixtureSource);
            runPortableManifest(catalog, counts);
            runWire(map(fixtureSource.loadJson(WIRE_FIXTURE), "wire fixture root"), counts);
            runSchema(map(fixtureSource.loadJson(SCHEMA_FIXTURE), "schema fixture root"), counts);
            runChain(map(fixtureSource.loadJson(CHAIN_FIXTURE), "chain fixture root"), counts);
            printSuccess(counts);
        } catch (Throwable failure) {
            System.err.println("G0 Java 17 reference conformance: FAIL");
            if (failure instanceof GovCjson.GovException protocolFailure) {
                System.err.println("protocol_code=" + protocolFailure.code());
            }
            System.err.println(failure.getClass().getSimpleName() + ": " + failure.getMessage());
            System.exit(1);
        }
    }

    /** 执行已由 ManifestCatalog 做过 byte/hash/canonical readback 的 direct public pages。 */
    private static void runPortableManifest(ManifestCatalog catalog, RunCounts counts) {
        counts.manifestIndexes++;
        counts.manifestAuthoringFixtures = catalog.authoringFixtureCount();
        for (ManifestCatalog.Page page : catalog.pages()) {
            switch (page.pageRole()) {
                case "wire" -> runManifestWirePage(page, counts);
                case "schema-positive", "schema-negative" -> runManifestSchemaPage(page, counts);
                case "chain" -> runManifestChainPage(page, counts);
                default -> throw new AssertionError("unsupported manifest page role: " + page.pageRole());
            }
            counts.manifestPages++;
        }
        for (ManifestCatalog.RawInput input : catalog.rawInputs()) {
            if (!"gov-cjson-parser".equals(input.inputKind())) {
                throw new AssertionError("unsupported manifest raw input kind: " + input.inputKind());
            }
            if (input.expectedCode() == 0) {
                expectSuccess(() -> GovCjson.parse(input.bytes()), "manifest raw " + input.name());
            } else {
                expectCode(() -> GovCjson.parse(input.bytes()), input.expectedCode(),
                        "manifest raw " + input.name());
            }
            counts.manifestRawInputs++;
        }
    }

    private static void runManifestWirePage(ManifestCatalog.Page page, RunCounts counts) {
        GovCjson.ObjectValue document = page.document();
        for (GovCjson.Value rawCase : govArray(document.value("wire_parser_cases"), "wire_parser_cases")) {
            GovCjson.ObjectValue testCase = govObject(rawCase, "wire parser case");
            long expected = govUnsigned(testCase.value("expected_code"), "wire parser expected_code");
            GovCjson.Bytes input = lowerHex(testCase.value("input_gov_cjson_hex"), "wire parser input");
            if (expected == 0) {
                expectSuccess(() -> GovCjson.parse(input), govText(testCase.value("name"), "wire parser name"));
            } else {
                expectCode(() -> GovCjson.parse(input), expected,
                        govText(testCase.value("name"), "wire parser name"));
            }
            counts.manifestWireParserCases++;
        }
        for (GovCjson.Value rawCase : govArray(document.value("wire_envelope_cases"), "wire_envelope_cases")) {
            GovCjson.ObjectValue testCase = govObject(rawCase, "wire envelope case");
            long expected = govUnsigned(testCase.value("expected_code"), "wire envelope expected_code");
            GovCjson.Bytes input = lowerHex(testCase.value("input_envelope_hex"), "wire envelope input");
            String name = govText(testCase.value("name"), "wire envelope name");
            if (expected == 0) {
                GovCjson.WireEnvelope envelope = parseEnvelopeSuccess(input, name);
                assertOptionalWireVectors(testCase, envelope, name);
            } else {
                expectCode(() -> GovCjson.parseEnvelope(input), expected, name);
            }
            counts.manifestWireEnvelopeCases++;
        }
        for (GovCjson.Value rawCase : govArray(
                document.value("wire_host_adapter_cases"), "wire_host_adapter_cases")) {
            GovCjson.ObjectValue testCase = govObject(rawCase, "wire host adapter case");
            GovCjson.ObjectValue payload = GovCjson.parse(lowerHex(
                    testCase.value("signed_payload_canonical_gov_cjson_hex"), "host adapter payload"));
            int expectedLength = checkedInt(govUnsigned(
                    testCase.value("expected_length"), "host adapter expected_length"),
                    "host adapter expected_length");
            int[] unsignedValues = unsignedIntArray(
                    testCase.value("unsigned_values"), "host adapter unsigned_values");
            long expected = govUnsigned(testCase.value("expected_code"), "host adapter expected_code");
            String name = govText(testCase.value("name"), "host adapter name");
            if (expected == 0) {
                expectSuccess(() -> GovCjson.encodeEnvelope(
                        payload, GovCjson.Bytes.fromUnsignedIntArray(unsignedValues, expectedLength)), name);
            } else {
                expectCode(() -> GovCjson.encodeEnvelope(
                        payload, GovCjson.Bytes.fromUnsignedIntArray(unsignedValues, expectedLength)),
                        expected, name);
            }
            counts.manifestWireAdapterCases++;
        }
        for (GovCjson.Value rawCase : govArray(
                document.value("wire_public_crypto_transport_cases"), "wire public crypto transport cases")) {
            GovCjson.ObjectValue testCase = govObject(rawCase, "wire public crypto transport case");
            requireEquals(GovCjson.ED25519_PUBLIC_KEY_BYTES,
                    lowerHex(testCase.value("public_key_hex"), "transport public key").length(),
                    "manifest transport public key length");
            requireEquals(GovCjson.ED25519_SIGNATURE_BYTES,
                    lowerHex(testCase.value("signature_hex"), "transport signature").length(),
                    "manifest transport signature length");
            lowerHex(testCase.value("message_hex"), "transport message");
            long verdict = govUnsigned(testCase.value("expected_verdict"), "transport expected verdict");
            require(verdict == GovCjson.VERDICT_INVALID || verdict == GovCjson.VERDICT_VALID,
                    "manifest transport verdict is outside 0|1");
            counts.manifestCryptoTransportCases++;
        }
        for (GovCjson.Value rawCase : govArray(
                document.value("document_precedence_cases"), "document precedence cases")) {
            GovCjson.ObjectValue testCase = govObject(rawCase, "document precedence case");
            String stage = govText(testCase.value("stage"), "document precedence stage");
            long expected = govUnsigned(testCase.value("expected_code"), "document precedence expected_code");
            String name = govText(testCase.value("name"), "document precedence name");
            Runnable operation = switch (stage) {
                case "physical" -> () -> GovCjson.parse(lowerHex(
                        testCase.value("input_gov_cjson_hex"), "physical input"));
                case "envelope", "hex", "common" -> () -> GovCjson.parseEnvelope(lowerHex(
                        testCase.value("input_envelope_hex"), "wire envelope input"));
                case "schema" -> () -> GovernanceSchema.parseDocument(lowerHex(
                        testCase.value("input_envelope_hex"), "schema envelope input"));
                default -> throw new AssertionError("unsupported document precedence stage: " + stage);
            };
            expectCode(operation, expected, name);
            counts.manifestPrecedenceCases++;
        }
    }

    private static void runManifestSchemaPage(ManifestCatalog.Page page, RunCounts counts) {
        for (GovCjson.Value rawCase : govArray(page.document().value("schema_cases"), "schema_cases")) {
            GovCjson.ObjectValue testCase = govObject(rawCase, "schema direct case");
            long expected = govUnsigned(testCase.value("expected_code"), "schema expected_code");
            GovCjson.Bytes input = lowerHex(testCase.value("input_envelope_hex"), "schema input");
            String name = govText(testCase.value("name"), "schema name");
            if (expected == 0) {
                GovernanceSchema.SchemaDocument document = parseSchemaSuccess(input, name);
                assertOptionalSchemaVectors(testCase, document, name);
            } else {
                expectCode(() -> GovernanceSchema.parseDocument(input), expected, name);
            }
            counts.manifestSchemaCases++;
        }
    }

    private static void runManifestChainPage(ManifestCatalog.Page page, RunCounts counts) {
        for (GovCjson.Value rawCase : govArray(page.document().value("chain_cases"), "chain_cases")) {
            GovCjson.ObjectValue testCase = govObject(rawCase, "chain direct case");
            List<GovCjson.Bytes> registries = hexCollection(
                    testCase.value("registry_envelopes_hex"), "chain registries");
            List<GovCjson.Bytes> revocations = hexCollection(
                    testCase.value("revocation_envelopes_hex"), "chain revocations");
            List<GovCjson.Bytes> declarations = hexCollection(
                    testCase.value("declaration_envelopes_hex"), "chain declarations");
            long expected = govUnsigned(testCase.value("expected_code"), "chain expected_code");
            String name = govText(testCase.value("name"), "chain name");
            if (expected == 0) {
                GovernanceChain.ChainHeads heads = parseChainSuccess(
                        registries, revocations, declarations, name);
                assertManifestChainHeadIdentities(testCase, heads, name);
            } else {
                expectCode(() -> GovernanceChain.validate(registries, revocations, declarations), expected, name);
            }
            counts.manifestChainCases++;
        }
    }

    private static GovCjson.WireEnvelope parseEnvelopeSuccess(GovCjson.Bytes input, String label) {
        try {
            return GovCjson.parseEnvelope(input);
        } catch (GovCjson.GovException failure) {
            throw new AssertionError(label + " unexpectedly failed with code=" + failure.code(), failure);
        }
    }

    private static GovernanceSchema.SchemaDocument parseSchemaSuccess(GovCjson.Bytes input, String label) {
        try {
            return GovernanceSchema.parseDocument(input);
        } catch (GovCjson.GovException failure) {
            throw new AssertionError(label + " unexpectedly failed with code=" + failure.code(), failure);
        }
    }

    private static GovernanceChain.ChainHeads parseChainSuccess(
            List<GovCjson.Bytes> registries,
            List<GovCjson.Bytes> revocations,
            List<GovCjson.Bytes> declarations,
            String label) {
        try {
            return GovernanceChain.validate(registries, revocations, declarations);
        } catch (GovCjson.GovException failure) {
            throw new AssertionError(label + " unexpectedly failed with code=" + failure.code(), failure);
        }
    }

    /** Direct corpus success cases pin all three derived heads, not merely their byte widths. */
    private static void assertManifestChainHeadIdentities(
            GovCjson.ObjectValue testCase, GovernanceChain.ChainHeads heads, String label) {
        List<GovCjson.Value> expected = govArray(
                testCase.value("expected_head_identities_sha256_hex"),
                label + " expected_head_identities_sha256_hex");
        requireEquals(3, expected.size(), label + " expected chain head count");
        requireEquals(manifestSha256Hex(expected.get(0), label + " registry head"),
                heads.registryHeadIdentity().toLowerHex(), label + " registry head identity");
        requireEquals(manifestSha256Hex(expected.get(1), label + " revocation head"),
                heads.revocationHeadIdentity().toLowerHex(), label + " revocation head identity");
        requireEquals(manifestSha256Hex(expected.get(2), label + " declaration head"),
                heads.declarationHeadIdentity().toLowerHex(), label + " declaration head identity");
    }

    private static String manifestSha256Hex(GovCjson.Value value, String label) {
        String expected = govText(value, label);
        requireEquals(64, expected.length(), label + " must be lowercase 64-hex SHA-256");
        GovCjson.Bytes decoded = GovCjson.Bytes.fromLowerHex(expected);
        requireEquals(32, decoded.length(), label + " must decode to 32 bytes");
        return expected;
    }

    private static void assertOptionalWireVectors(
            GovCjson.ObjectValue testCase, GovCjson.WireEnvelope envelope, String label) {
        assertOptionalHex(testCase, "canonical_signed_payload_hex",
                envelope.canonicalSignedPayload().toLowerHex(), label);
        assertOptionalHex(testCase, "domain_prefix_hex", envelope.domainPrefix().toLowerHex(), label);
        assertOptionalHex(testCase, "message_hex", envelope.message().toLowerHex(), label);
        assertOptionalHex(testCase, "document_identity_sha256_hex",
                envelope.documentIdentity().toLowerHex(), label);
    }

    private static void assertOptionalSchemaVectors(
            GovCjson.ObjectValue testCase, GovernanceSchema.SchemaDocument document, String label) {
        assertOptionalHex(testCase, "canonical_signed_payload_hex",
                document.canonicalSignedPayload().toLowerHex(), label);
        assertOptionalHex(testCase, "domain_prefix_hex", document.domainPrefix().toLowerHex(), label);
        assertOptionalHex(testCase, "message_hex", document.message().toLowerHex(), label);
        assertOptionalHex(testCase, "document_identity_sha256_hex",
                document.documentIdentity().toLowerHex(), label);
    }

    private static void assertOptionalHex(
            GovCjson.ObjectValue testCase, String field, String actual, String label) {
        GovCjson.Value expected = testCase.value(field);
        if (expected != null) {
            requireEquals(govText(expected, field), actual, label + " " + field);
        }
    }

    private static List<GovCjson.Bytes> hexCollection(GovCjson.Value value, String label) {
        List<GovCjson.Bytes> result = new ArrayList<>();
        for (GovCjson.Value item : govArray(value, label)) {
            result.add(lowerHex(item, label + " item"));
        }
        return result;
    }

    private static int[] unsignedIntArray(GovCjson.Value value, String label) {
        List<GovCjson.Value> rawValues = govArray(value, label);
        int[] result = new int[rawValues.size()];
        for (int index = 0; index < result.length; index++) {
            long raw = govUnsigned(rawValues.get(index), label + "[" + index + "]");
            if (raw > Integer.MAX_VALUE) {
                throw new GovCjson.GovException(GovCjson.REJECT_BYTE_TUPLE,
                        label + "[" + index + "] cannot be represented by the int[] adapter");
            }
            result[index] = (int) raw;
        }
        return result;
    }

    private static int checkedInt(long value, String label) {
        if (value > Integer.MAX_VALUE) {
            throw new AssertionError(label + " exceeds Java int adapter range");
        }
        return (int) value;
    }

    private static void runWire(Map<String, Object> fixture, RunCounts counts) {
        requireEquals(GovCjson.PROFILE, text(fixture.get("profile"), "wire profile"), "wire profile");
        for (Object rawCase : list(fixture.get("reference_cases"), "wire reference_cases")) {
            Map<String, Object> reference = map(rawCase, "wire reference case");
            GovCjson.Bytes physical = physicalForCase(reference);
            GovCjson.WireEnvelope envelope = GovCjson.parseEnvelope(physical);
            requireEquals(text(reference.get("envelope_hex"), "wire envelope_hex"),
                    physical.toLowerHex(), "wire envelope bytes");
            requireEquals(text(reference.get("canonical_signed_payload_hex"), "wire canonical payload"),
                    envelope.canonicalSignedPayload().toLowerHex(), "wire canonical signed payload");
            requireEquals(text(reference.get("domain_prefix_hex"), "wire domain prefix"),
                    envelope.domainPrefix().toLowerHex(), "wire domain prefix");
            requireEquals(text(reference.get("message_hex"), "wire message"),
                    envelope.message().toLowerHex(), "wire message");
            requireEquals(text(reference.get("document_identity_sha256_hex"), "wire identity"),
                    envelope.documentIdentity().toLowerHex(), "wire identity");
            requireEquals(GovCjson.STATUS_REFERENCE_ONLY, envelope.status(), "wire reference-only status");
            counts.wirePositive++;
        }
        for (Object rawCase : list(fixture.get("syntax_rejections"), "wire syntax_rejections")) {
            Map<String, Object> rejection = map(rawCase, "wire syntax rejection");
            GovCjson.Bytes malformed = GovCjson.Bytes.fromLowerHex(
                    text(rejection.get("payload_hex"), "wire rejection payload_hex"));
            expectCode(
                    () -> GovCjson.parse(malformed),
                    integer(rejection.get("error_code"), "wire rejection code"),
                    text(rejection.get("name"), "wire rejection name"));
            counts.wireRejected++;
        }
        for (Object rawVector : list(fixture.get("ed25519_public_vectors"), "ed25519 public vectors")) {
            Map<String, Object> vector = map(rawVector, "ed25519 public vector");
            requireEquals(GovCjson.ED25519_PUBLIC_KEY_BYTES,
                    GovCjson.Bytes.fromLowerHex(text(vector.get("public_key_hex"), "public_key_hex")).length(),
                    "ed25519 public key length");
            requireEquals(GovCjson.ED25519_SIGNATURE_BYTES,
                    GovCjson.Bytes.fromLowerHex(text(vector.get("signature_hex"), "signature_hex")).length(),
                    "ed25519 signature length");
            GovCjson.Bytes.fromLowerHex(text(vector.get("message_hex"), "message_hex"));
            long expectedVerdict = integer(vector.get("expected_verdict"), "expected_verdict");
            require(expectedVerdict == GovCjson.VERDICT_INVALID || expectedVerdict == GovCjson.VERDICT_VALID,
                    "ed25519 fixture verdict is outside 0|1");
            counts.ed25519TransportOnly++;
        }
    }

    private static void runSchema(Map<String, Object> fixture, RunCounts counts) {
        requireEquals(GovCjson.PROFILE, text(fixture.get("profile"), "schema profile"), "schema profile");
        requireEquals(GovCjson.STATUS_REFERENCE_ONLY,
                text(fixture.get("schema_reference_status"), "schema reference status"),
                "schema reference-only status");
        requireEquals("ZERO_BYTES_UNVERIFIED",
                text(fixture.get("signature_semantics"), "schema signature semantics"),
                "schema signature semantics");

        Map<String, Map<String, Object>> casesByName = new LinkedHashMap<>();
        for (Object rawCase : list(fixture.get("reference_cases"), "schema reference_cases")) {
            Map<String, Object> reference = map(rawCase, "schema reference case");
            casesByName.put(text(reference.get("name"), "schema case name"), reference);
            GovCjson.Bytes physical = physicalForCase(reference);
            GovernanceSchema.SchemaDocument document = GovernanceSchema.parseDocument(physical);
            Map<String, Object> payload = map(reference.get("signed_payload"), "schema signed_payload");
            String kind = text(payload.get("kind"), "schema kind");
            GovernanceSchema.SchemaDocument specific = switch (kind) {
                case GovCjson.ROOT_REGISTRY -> GovernanceSchema.parseRootRegistry(physical);
                case GovCjson.REVOCATION_SNAPSHOT -> GovernanceSchema.parseRevocationSnapshot(physical);
                case GovCjson.SOURCE_SNAPSHOT_DECLARATION ->
                        GovernanceSchema.parseSourceSnapshotDeclaration(physical);
                default -> throw new AssertionError("schema fixture has an unsupported kind");
            };
            requireEquals(document.documentIdentity(), specific.documentIdentity(), "schema specific parser identity");
            requireEquals(text(reference.get("canonical_signed_payload_hex"), "schema canonical payload"),
                    document.canonicalSignedPayload().toLowerHex(), "schema canonical signed payload");
            requireEquals(text(reference.get("domain_prefix_hex"), "schema domain prefix"),
                    document.domainPrefix().toLowerHex(), "schema domain prefix");
            requireEquals(text(reference.get("document_identity_sha256_hex"), "schema identity"),
                    document.documentIdentity().toLowerHex(), "schema identity");
            requireEquals(kind, document.kind(), "schema kind");
            requireEquals(text(payload.get("key_id"), "schema key_id"), document.keyId(), "schema key_id");
            requireEquals(integer(payload.get("sequence"), "schema sequence"), document.sequence(),
                    "schema sequence");
            requireEquals(GovernanceSchema.STATUS_REFERENCE_ONLY, document.status(),
                    "schema reference-only document status");
            counts.schemaPositive++;
        }
        for (Object rawRejection : list(fixture.get("schema_rejections"), "schema_rejections")) {
            Map<String, Object> rejection = map(rawRejection, "schema rejection");
            String baseName = text(rejection.get("base_case"), "schema rejection base_case");
            Map<String, Object> base = casesByName.get(baseName);
            require(base != null, "schema rejection references an unknown base case");
            Map<String, Object> payload = deepMap(map(base.get("signed_payload"), "schema base payload"));
            applySchemaMutation(payload, rejection);
            GovCjson.Bytes physical = GovCjson.encodeEnvelope(
                    toGovObject(payload),
                    GovCjson.Bytes.fromLowerHex(text(base.get("signature_hex"), "schema signature_hex")));
            expectCode(
                    () -> GovernanceSchema.parseDocument(physical),
                    integer(rejection.get("error_code"), "schema rejection code"),
                    text(rejection.get("name"), "schema rejection name"));
            counts.schemaRejected++;
        }
    }

    private static void runChain(Map<String, Object> fixture, RunCounts counts) {
        requireEquals(GovCjson.PROFILE, text(fixture.get("profile"), "chain profile"), "chain profile");
        requireEquals(GovernanceChain.STATUS_REFERENCE_ONLY,
                text(fixture.get("chain_shape_reference_status"), "chain reference status"),
                "chain reference-only status");
        requireEquals("ZERO_BYTES_UNVERIFIED",
                text(fixture.get("signature_semantics"), "chain signature semantics"),
                "chain signature semantics");

        Map<String, Object> referenceCollections = map(
                fixture.get("reference_collections"), "reference_collections");
        Map<String, Object> copiedCollections = deepMap(referenceCollections);
        List<GovCjson.Bytes> registries = encodeCollection(list(copiedCollections.get("registry"), "registry"));
        List<GovCjson.Bytes> revocations = encodeCollection(list(copiedCollections.get("revocation"), "revocation"));
        List<GovCjson.Bytes> declarations = encodeCollection(list(copiedCollections.get("declaration"), "declaration"));
        Collections.reverse(registries);
        Collections.reverse(revocations);
        Collections.reverse(declarations);
        GovernanceChain.ChainHeads heads = GovernanceChain.validate(registries, revocations, declarations);
        List<Object> expectedHeads = list(fixture.get("expected_head_identities_sha256_hex"), "chain heads");
        requireEquals(3, expectedHeads.size(), "chain head count");
        requireEquals(text(expectedHeads.get(0), "registry head"), heads.registryHeadIdentity().toLowerHex(),
                "registry head identity");
        requireEquals(text(expectedHeads.get(1), "revocation head"), heads.revocationHeadIdentity().toLowerHex(),
                "revocation head identity");
        requireEquals(text(expectedHeads.get(2), "declaration head"), heads.declarationHeadIdentity().toLowerHex(),
                "declaration head identity");
        requireEquals(GovernanceChain.STATUS_REFERENCE_ONLY, heads.status(), "chain reference-only status");
        counts.chainPositive++;

        for (Object rawRejection : list(fixture.get("chain_shape_rejections"), "chain rejections")) {
            Map<String, Object> rejection = map(rawRejection, "chain rejection");
            Map<String, Object> mutatedCollections = deepMap(referenceCollections);
            applyChainMutation(mutatedCollections, rejection);
            List<GovCjson.Bytes> mutatedRegistries = encodeCollection(
                    list(mutatedCollections.get("registry"), "mutated registry"));
            List<GovCjson.Bytes> mutatedRevocations = encodeCollection(
                    list(mutatedCollections.get("revocation"), "mutated revocation"));
            List<GovCjson.Bytes> mutatedDeclarations = encodeCollection(
                    list(mutatedCollections.get("declaration"), "mutated declaration"));
            expectCode(
                    () -> GovernanceChain.validate(mutatedRegistries, mutatedRevocations, mutatedDeclarations),
                    integer(rejection.get("error_code"), "chain rejection code"),
                    text(rejection.get("name"), "chain rejection name"));
            counts.chainRejected++;
        }
    }

    /** 优先直接消费未来新增的 envelope_hex；当前 fixture 没有时才由公开 raw values 重建。 */
    private static GovCjson.Bytes physicalForCase(Map<String, Object> reference) {
        Object directEnvelope = reference.get("envelope_hex");
        if (directEnvelope instanceof String directHex) {
            return GovCjson.Bytes.fromLowerHex(directHex);
        }
        return GovCjson.encodeEnvelope(
                toGovObject(map(reference.get("signed_payload"), "signed_payload")),
                GovCjson.Bytes.fromLowerHex(text(reference.get("signature_hex"), "signature_hex")));
    }

    private static List<GovCjson.Bytes> encodeCollection(List<Object> rawDocuments) {
        List<GovCjson.Bytes> result = new ArrayList<>();
        for (Object rawDocument : rawDocuments) {
            Map<String, Object> document = map(rawDocument, "chain document");
            Object directEnvelope = document.get("envelope_hex");
            if (directEnvelope instanceof String directHex) {
                result.add(GovCjson.Bytes.fromLowerHex(directHex));
            } else {
                result.add(GovCjson.encodeEnvelope(toGovObject(document), ZERO_SIGNATURE));
            }
        }
        return result;
    }

    private static void applySchemaMutation(Map<String, Object> payload, Map<String, Object> rejection) {
        PathParent target = pathParent(payload, list(rejection.get("path"), "schema mutation path"));
        String operation = text(rejection.get("operation"), "schema mutation operation");
        switch (operation) {
            case "set", "add" -> setChild(target.parent(), target.leaf(), deepCopy(rejection.get("value")));
            case "drop" -> {
                Map<String, Object> parent = map(target.parent(), "schema drop parent");
                parent.remove(text(target.leaf(), "schema drop field"));
            }
            case "swap" -> {
                Object child = child(target.parent(), target.leaf());
                List<Object> targetArray = list(child, "schema swap target");
                List<Object> indexes = list(rejection.get("indexes"), "schema swap indexes");
                requireEquals(2, indexes.size(), "schema swap index count");
                int left = index(indexes.get(0), "schema swap left");
                int right = index(indexes.get(1), "schema swap right");
                Object temporary = targetArray.get(left);
                targetArray.set(left, targetArray.get(right));
                targetArray.set(right, temporary);
            }
            default -> throw new AssertionError("unsupported schema mutation operation: " + operation);
        }
    }

    private static void applyChainMutation(Map<String, Object> collections, Map<String, Object> rejection) {
        String collectionName = text(rejection.get("collection"), "chain mutation collection");
        int documentIndex = index(rejection.get("index"), "chain mutation index");
        List<Object> collection = list(collections.get(collectionName), "chain mutation collection values");
        PathParent target = pathParent(map(collection.get(documentIndex), "chain mutation document"),
                list(rejection.get("path"), "chain mutation path"));
        String operation = text(rejection.get("operation"), "chain mutation operation");
        if ("set".equals(operation)) {
            setChild(target.parent(), target.leaf(), deepCopy(rejection.get("value")));
            return;
        }
        if ("copy".equals(operation)) {
            String sourceCollection = text(rejection.get("source_collection"), "chain source collection");
            int sourceIndex = index(rejection.get("source_index"), "chain source index");
            Map<String, Object> sourceDocument = map(
                    list(collections.get(sourceCollection), "chain source collection values").get(sourceIndex),
                    "chain source document");
            PathParent source = pathParent(sourceDocument,
                    list(rejection.get("source_path"), "chain source path"));
            setChild(target.parent(), target.leaf(), deepCopy(child(source.parent(), source.leaf())));
            return;
        }
        throw new AssertionError("unsupported chain mutation operation: " + operation);
    }

    private static PathParent pathParent(Object root, List<Object> path) {
        require(!path.isEmpty(), "fixture path must not be empty");
        Object current = root;
        for (int position = 0; position < path.size() - 1; position++) {
            current = child(current, path.get(position));
        }
        return new PathParent(current, path.get(path.size() - 1));
    }

    private static Object child(Object parent, Object leaf) {
        if (parent instanceof Map<?, ?>) {
            return map(parent, "fixture map parent").get(text(leaf, "fixture map field"));
        }
        if (parent instanceof List<?>) {
            return list(parent, "fixture list parent").get(index(leaf, "fixture list index"));
        }
        throw new AssertionError("fixture path parent is neither map nor list");
    }

    private static void setChild(Object parent, Object leaf, Object value) {
        if (parent instanceof Map<?, ?>) {
            map(parent, "fixture map parent").put(text(leaf, "fixture map field"), value);
            return;
        }
        if (parent instanceof List<?>) {
            list(parent, "fixture list parent").set(index(leaf, "fixture list index"), value);
            return;
        }
        throw new AssertionError("fixture mutation parent is neither map nor list");
    }

    private static GovCjson.ObjectValue govObject(GovCjson.Value value, String label) {
        if (value instanceof GovCjson.ObjectValue result) {
            return result;
        }
        throw new AssertionError(label + " must be a GOV-CJSON object");
    }

    private static List<GovCjson.Value> govArray(GovCjson.Value value, String label) {
        if (value instanceof GovCjson.ArrayValue result) {
            return result.values();
        }
        throw new AssertionError(label + " must be a GOV-CJSON array");
    }

    private static String govText(GovCjson.Value value, String label) {
        if (value instanceof GovCjson.TextValue result) {
            return result.value();
        }
        throw new AssertionError(label + " must be GOV-CJSON text");
    }

    private static long govUnsigned(GovCjson.Value value, String label) {
        if (value instanceof GovCjson.UIntValue result && result.value() >= 0) {
            return result.value();
        }
        throw new AssertionError(label + " must be GOV-CJSON u63");
    }

    private static GovCjson.Bytes lowerHex(GovCjson.Value value, String label) {
        return GovCjson.Bytes.fromLowerHex(govText(value, label));
    }

    private static GovCjson.ObjectValue toGovObject(Map<String, Object> source) {
        Map<String, GovCjson.Value> result = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : source.entrySet()) {
            result.put(entry.getKey(), toGovValue(entry.getValue()));
        }
        return new GovCjson.ObjectValue(result);
    }

    private static GovCjson.Value toGovValue(Object source) {
        if (source instanceof String text) {
            return new GovCjson.TextValue(text);
        }
        if (source instanceof Long integer) {
            return new GovCjson.UIntValue(integer);
        }
        if (source instanceof Map<?, ?>) {
            return toGovObject(map(source, "fixture GOV-CJSON object"));
        }
        if (source instanceof List<?>) {
            List<GovCjson.Value> values = new ArrayList<>();
            for (Object item : list(source, "fixture GOV-CJSON array")) {
                values.add(toGovValue(item));
            }
            return new GovCjson.ArrayValue(values);
        }
        throw new AssertionError("fixture value cannot enter GOV-CJSON core");
    }

    private static Map<String, Object> deepMap(Map<String, Object> source) {
        @SuppressWarnings("unchecked")
        Map<String, Object> result = (Map<String, Object>) deepCopy(source);
        return result;
    }

    private static Object deepCopy(Object source) {
        if (source instanceof Map<?, ?>) {
            Map<String, Object> result = new LinkedHashMap<>();
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) source).entrySet()) {
                result.put(text(entry.getKey(), "fixture map key"), deepCopy(entry.getValue()));
            }
            return result;
        }
        if (source instanceof List<?>) {
            List<Object> result = new ArrayList<>();
            for (Object item : (List<?>) source) {
                result.add(deepCopy(item));
            }
            return result;
        }
        if (source == null || source instanceof String || source instanceof Long || source instanceof Boolean) {
            return source;
        }
        throw new AssertionError("fixture contains an unsupported value type");
    }

    private static Map<String, Object> map(Object value, String label) {
        if (!(value instanceof Map<?, ?> raw)) {
            throw new AssertionError(label + " must be a map");
        }
        for (Object key : raw.keySet()) {
            text(key, label + " key");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> result = (Map<String, Object>) raw;
        return result;
    }

    private static List<Object> list(Object value, String label) {
        if (!(value instanceof List<?> raw)) {
            throw new AssertionError(label + " must be a list");
        }
        @SuppressWarnings("unchecked")
        List<Object> result = (List<Object>) raw;
        return result;
    }

    private static String text(Object value, String label) {
        if (!(value instanceof String result)) {
            throw new AssertionError(label + " must be a string");
        }
        return result;
    }

    private static long integer(Object value, String label) {
        if (!(value instanceof Long result)) {
            throw new AssertionError(label + " must be an integer");
        }
        return result;
    }

    private static int index(Object value, String label) {
        long result = integer(value, label);
        if (result < 0 || result > Integer.MAX_VALUE) {
            throw new AssertionError(label + " is outside index range");
        }
        return (int) result;
    }

    private static void expectCode(Runnable action, long expected, String label) {
        long actual = captureCode(action, label);
        requireEquals(expected, actual, label + " protocol code");
    }

    private static void expectSuccess(Runnable action, String label) {
        try {
            action.run();
        } catch (GovCjson.GovException failure) {
            throw new AssertionError(label + " unexpectedly failed with code=" + failure.code(), failure);
        }
    }

    private static int captureCode(Runnable action, String label) {
        try {
            action.run();
        } catch (GovCjson.GovException failure) {
            return failure.code();
        }
        throw new AssertionError(label + " unexpectedly succeeded");
    }

    private static void require(boolean condition, String label) {
        if (!condition) {
            throw new AssertionError(label);
        }
    }

    private static void requireEquals(Object expected, Object actual, String label) {
        if (!java.util.Objects.equals(expected, actual)) {
            throw new AssertionError(label + " expected=" + expected + " actual=" + actual);
        }
    }

    private static void printSuccess(RunCounts counts) {
        System.out.println("G0 Java 17 reference conformance: PASS");
        System.out.println("status=REFERENCE_ONLY");
        System.out.println("manifest.indexes=" + counts.manifestIndexes);
        System.out.println("manifest.authoring_fixtures=" + counts.manifestAuthoringFixtures);
        System.out.println("manifest.pages=" + counts.manifestPages);
        System.out.println("manifest.raw_inputs=" + counts.manifestRawInputs);
        System.out.println("manifest.wire_parser_cases=" + counts.manifestWireParserCases);
        System.out.println("manifest.wire_envelope_cases=" + counts.manifestWireEnvelopeCases);
        System.out.println("manifest.wire_adapter_cases=" + counts.manifestWireAdapterCases);
        System.out.println("manifest.crypto_transport_only=" + counts.manifestCryptoTransportCases
                + " (no verifier implemented or invoked)");
        System.out.println("manifest.precedence_cases=" + counts.manifestPrecedenceCases);
        System.out.println("manifest.schema_cases=" + counts.manifestSchemaCases);
        System.out.println("manifest.chain_cases=" + counts.manifestChainCases);
        System.out.println("wire.positive=" + counts.wirePositive);
        System.out.println("wire.fail_closed=" + counts.wireRejected);
        System.out.println("wire.ed25519_transport_only=" + counts.ed25519TransportOnly
                + " (no verifier implemented or invoked)");
        System.out.println("schema.positive=" + counts.schemaPositive);
        System.out.println("schema.fail_closed=" + counts.schemaRejected);
        System.out.println("chain.positive=" + counts.chainPositive);
        System.out.println("chain.fail_closed=" + counts.chainRejected);
    }

    private record PathParent(Object parent, Object leaf) {
    }

    private static final class RunCounts {
        private int wirePositive;
        private int wireRejected;
        private int ed25519TransportOnly;
        private int schemaPositive;
        private int schemaRejected;
        private int chainPositive;
        private int chainRejected;
        private int manifestIndexes;
        private int manifestAuthoringFixtures;
        private int manifestPages;
        private int manifestRawInputs;
        private int manifestWireParserCases;
        private int manifestWireEnvelopeCases;
        private int manifestWireAdapterCases;
        private int manifestCryptoTransportCases;
        private int manifestPrecedenceCases;
        private int manifestSchemaCases;
        private int manifestChainCases;
    }
}
