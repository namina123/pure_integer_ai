package org.pidslca.g0;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 分层 portable conformance corpus 的测试 catalog adapter。
 *
 * <p>index 与每个 page 都先经过 {@link GovCjson#parse(GovCjson.Bytes)} 的 canonical
 * readback；sidecar、authoring fixture、page 与 raw artifact 的 byte count/SHA-256 都在
 * 本 adapter 核对。文件系统和 manifest 仅属于测试输入边界，绝不进入 portable core。</p>
 */
final class ManifestCatalog {
    static final String INDEX_FILE = "gov_g0_portable_conformance_manifest_v1.json";
    private static final String INDEX_SIDECAR_FILE = "gov_g0_portable_conformance_manifest_v1.sha256";
    private static final String EXPECTED_INDEX_SHA256 =
            "6868186cf3ae4948a5940e633251132bb9094ad72a1e7cbda5b7338c3e069521";
    private static final String INDEX_KIND = "GOVERNANCE_PORTABLE_CONFORMANCE_MANIFEST_INDEX_V1";
    private static final Set<String> INDEX_FIELDS = Set.of(
            "artifact_kind", "authoring_fixture_sha256", "page_order", "profile",
            "raw_input_artifacts", "version");
    private static final Set<String> AUTHORING_FIXTURE_FIELDS = Set.of("file_name", "sha256_hex");
    private static final Set<String> PAGE_ENTRY_FIELDS = Set.of(
            "byte_count", "file_name", "page_role", "sha256_hex");
    private static final Set<String> RAW_ENTRY_FIELDS = Set.of(
            "byte_count", "expected_code", "file_name", "input_kind", "name", "sha256_hex");
    private static final Set<String> WIRE_PAGE_FIELDS = Set.of(
            "document_precedence_cases", "page_role", "profile", "version",
            "wire_envelope_cases", "wire_host_adapter_cases", "wire_parser_cases",
            "wire_public_crypto_transport_cases");
    private static final Set<String> SCHEMA_PAGE_FIELDS = Set.of(
            "page_role", "profile", "schema_cases", "version");
    private static final Set<String> CHAIN_PAGE_FIELDS = Set.of(
            "chain_cases", "page_role", "profile", "version");

    private final List<Page> pages;
    private final List<RawInput> rawInputs;
    private final int authoringFixtureCount;

    private ManifestCatalog(List<Page> pages, List<RawInput> rawInputs, int authoringFixtureCount) {
        this.pages = List.copyOf(pages);
        this.rawInputs = List.copyOf(rawInputs);
        this.authoringFixtureCount = authoringFixtureCount;
    }

    static ManifestCatalog load(PublicFixtureSource source) {
        GovCjson.Bytes indexBytes = source.loadBytes(INDEX_FILE);
        GovCjson.ObjectValue index = GovCjson.parse(indexBytes);
        verifyIndexSidecar(source, indexBytes);
        requireExact(index, INDEX_FIELDS, "manifest index");
        requireEquals(INDEX_KIND, text(index.value("artifact_kind"), "manifest artifact_kind"),
                "manifest artifact_kind");
        requireEquals(GovCjson.PROFILE, text(index.value("profile"), "manifest profile"),
                "manifest profile");
        requireEquals(1L, unsigned(index.value("version"), "manifest version"), "manifest version");

        List<GovCjson.Value> authoringFixtures = array(
                index.value("authoring_fixture_sha256"), "authoring_fixture_sha256");
        requireEquals(3, authoringFixtures.size(), "manifest authoring fixture count");
        verifyAuthoringFixtures(source, authoringFixtures);
        List<GovCjson.Value> pageEntries = array(index.value("page_order"), "page_order");
        List<GovCjson.Value> rawEntries = array(index.value("raw_input_artifacts"), "raw_input_artifacts");
        requireEquals(13, pageEntries.size(), "manifest page count");
        requireEquals(8, rawEntries.size(), "manifest raw input count");
        List<Page> pages = loadPages(source, pageEntries);
        List<RawInput> rawInputs = loadRawInputs(source, rawEntries);
        return new ManifestCatalog(pages, rawInputs, authoringFixtures.size());
    }

    List<Page> pages() {
        return pages;
    }

    List<RawInput> rawInputs() {
        return rawInputs;
    }

    int authoringFixtureCount() {
        return authoringFixtureCount;
    }

    private static void verifyIndexSidecar(PublicFixtureSource source, GovCjson.Bytes indexBytes) {
        String actualHash = GovCjson.sha256(indexBytes).toLowerHex();
        requireEquals(EXPECTED_INDEX_SHA256, actualHash, "pinned manifest index SHA-256");
        GovCjson.Bytes sidecar = source.loadBytes(INDEX_SIDECAR_FILE);
        String expectedSidecar = EXPECTED_INDEX_SHA256 + "  " + INDEX_FILE;
        requireEquals(expectedSidecar, ascii(sidecar, "manifest sidecar"), "manifest sidecar");
    }

    private static void verifyAuthoringFixtures(
            PublicFixtureSource source, List<GovCjson.Value> entries) {
        Set<String> names = new HashSet<>();
        for (GovCjson.Value rawEntry : entries) {
            GovCjson.ObjectValue entry = object(rawEntry, "authoring fixture entry");
            requireExact(entry, AUTHORING_FIXTURE_FIELDS, "authoring fixture entry");
            String fileName = requireBareFixtureName(
                    entry.value("file_name"), "authoring fixture file_name");
            String expectedHash = sha256Hex(entry.value("sha256_hex"), "authoring fixture sha256_hex");
            require(names.add(fileName), "authoring fixture file_name is duplicated");
            requireEquals(expectedHash, GovCjson.sha256(source.loadBytes(fileName)).toLowerHex(),
                    "authoring fixture SHA-256 for " + fileName);
        }
    }

    private static List<Page> loadPages(PublicFixtureSource source, List<GovCjson.Value> entries) {
        List<Page> result = new ArrayList<>();
        Set<String> names = new HashSet<>();
        for (GovCjson.Value rawEntry : entries) {
            GovCjson.ObjectValue entry = object(rawEntry, "manifest page entry");
            requireExact(entry, PAGE_ENTRY_FIELDS, "manifest page entry");
            String fileName = requireBareFixtureName(entry.value("file_name"), "manifest page file_name");
            String role = text(entry.value("page_role"), "manifest page_role");
            long byteCount = unsigned(entry.value("byte_count"), "manifest page byte_count");
            String expectedHash = sha256Hex(entry.value("sha256_hex"), "manifest page sha256_hex");
            require(names.add(fileName), "manifest page file_name is duplicated");
            GovCjson.Bytes content = verifyArtifact(source, fileName, byteCount, expectedHash);
            GovCjson.ObjectValue page = GovCjson.parse(content);
            verifyPageShape(page, role, fileName);
            requireEquals(GovCjson.PROFILE, text(page.value("profile"), "page profile"),
                    "page profile for " + fileName);
            requireEquals(1L, unsigned(page.value("version"), "page version"),
                    "page version for " + fileName);
            requireEquals(role, text(page.value("page_role"), "page page_role"),
                    "page role for " + fileName);
            result.add(new Page(fileName, role, page));
        }
        return result;
    }

    private static void verifyPageShape(GovCjson.ObjectValue page, String role, String fileName) {
        Set<String> expected = switch (role) {
            case "wire" -> WIRE_PAGE_FIELDS;
            case "schema-positive", "schema-negative" -> SCHEMA_PAGE_FIELDS;
            case "chain" -> CHAIN_PAGE_FIELDS;
            default -> throw new IllegalArgumentException("unregistered manifest page_role for " + fileName);
        };
        requireExact(page, expected, "manifest page " + fileName);
    }

    private static List<RawInput> loadRawInputs(PublicFixtureSource source, List<GovCjson.Value> entries) {
        List<RawInput> result = new ArrayList<>();
        Set<String> names = new HashSet<>();
        for (GovCjson.Value rawEntry : entries) {
            GovCjson.ObjectValue entry = object(rawEntry, "manifest raw input entry");
            requireExact(entry, RAW_ENTRY_FIELDS, "manifest raw input entry");
            String fileName = requireBareFixtureName(entry.value("file_name"), "raw input file_name");
            String name = text(entry.value("name"), "raw input name");
            String inputKind = text(entry.value("input_kind"), "raw input input_kind");
            long byteCount = unsigned(entry.value("byte_count"), "raw input byte_count");
            long expectedCode = unsigned(entry.value("expected_code"), "raw input expected_code");
            String expectedHash = sha256Hex(entry.value("sha256_hex"), "raw input sha256_hex");
            require(names.add(fileName), "raw input file_name is duplicated");
            result.add(new RawInput(
                    fileName, name, inputKind, expectedCode,
                    verifyArtifact(source, fileName, byteCount, expectedHash)));
        }
        return result;
    }

    private static GovCjson.Bytes verifyArtifact(
            PublicFixtureSource source, String fileName, long expectedByteCount, String expectedHash) {
        GovCjson.Bytes content = source.loadBytes(fileName);
        requireEquals(expectedByteCount, (long) content.length(), "artifact byte_count for " + fileName);
        requireEquals(expectedHash, GovCjson.sha256(content).toLowerHex(),
                "artifact SHA-256 for " + fileName);
        return content;
    }

    private static String sha256Hex(GovCjson.Value value, String label) {
        String result = text(value, label);
        GovCjson.Bytes decoded = GovCjson.Bytes.fromLowerHex(result);
        if (decoded.length() != 32 || result.length() != 64) {
            throw new IllegalArgumentException(label + " is not a lowercase SHA-256");
        }
        return result;
    }

    /** manifest 只允许固定 basename，不能把 host-relative path 带入 artifact identity。 */
    private static String requireBareFixtureName(GovCjson.Value value, String label) {
        String result = text(value, label);
        if (result.isEmpty() || result.charAt(0) == '.' || result.indexOf('/') >= 0
                || result.indexOf('\\') >= 0 || ".".equals(result) || "..".equals(result)) {
            throw new IllegalArgumentException(label + " is not a bare fixture name");
        }
        for (int index = 0; index < result.length(); index++) {
            char character = result.charAt(index);
            if (!((character >= 'a' && character <= 'z')
                    || (character >= 'A' && character <= 'Z')
                    || (character >= '0' && character <= '9')
                    || character == '.' || character == '_' || character == '-')) {
                throw new IllegalArgumentException(label + " contains an invalid basename character");
            }
        }
        return result;
    }

    private static String ascii(GovCjson.Bytes value, String label) {
        char[] result = new char[value.length()];
        for (int index = 0; index < result.length; index++) {
            int item = value.unsignedAt(index);
            if (item > 0x7f) {
                throw new IllegalArgumentException(label + " contains non-ASCII bytes");
            }
            result[index] = (char) item;
        }
        return new String(result);
    }

    private static GovCjson.ObjectValue object(GovCjson.Value value, String label) {
        if (value instanceof GovCjson.ObjectValue result) {
            return result;
        }
        throw new IllegalArgumentException(label + " must be an object");
    }

    private static List<GovCjson.Value> array(GovCjson.Value value, String label) {
        if (value instanceof GovCjson.ArrayValue result) {
            return result.values();
        }
        throw new IllegalArgumentException(label + " must be an array");
    }

    private static String text(GovCjson.Value value, String label) {
        if (value instanceof GovCjson.TextValue result) {
            return result.value();
        }
        throw new IllegalArgumentException(label + " must be text");
    }

    private static long unsigned(GovCjson.Value value, String label) {
        if (value instanceof GovCjson.UIntValue result && result.value() >= 0) {
            return result.value();
        }
        throw new IllegalArgumentException(label + " must be u63");
    }

    private static void requireExact(GovCjson.ObjectValue value, Set<String> fields, String label) {
        if (!value.values().keySet().equals(fields)) {
            throw new IllegalArgumentException(label + " field set is not exact");
        }
    }

    private static void requireEquals(Object expected, Object actual, String label) {
        if (!java.util.Objects.equals(expected, actual)) {
            throw new IllegalArgumentException(label + " expected=" + expected + " actual=" + actual);
        }
    }

    private static void require(boolean condition, String label) {
        if (!condition) {
            throw new IllegalArgumentException(label);
        }
    }

    record Page(String fileName, String pageRole, GovCjson.ObjectValue document) {
    }

    record RawInput(
            String fileName,
            String name,
            String inputKind,
            long expectedCode,
            GovCjson.Bytes bytes) {
    }
}
