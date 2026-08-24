package org.pidslca.g0;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * GOV-CJSON-1 的 Java 17 可移植 reference core。
 *
 * <p>本类只处理有限 ASCII 字节、u63、固定 domain message 与 SHA-256 identity。它不读取
 * 文件、环境或网络，不执行 Ed25519 验证，也不产生任何信任、资格或 capability 结论。</p>
 */
public final class GovCjson {
    public static final String PROFILE = "GOV-CJSON-1";
    public static final String STATUS_REFERENCE_ONLY = "PORTABILITY_CONTRACT_REFERENCE_ONLY";

    public static final int MAX_DOCUMENT_BYTES = 65_536;
    public static final int MAX_DEPTH = 16;
    public static final int MAX_OBJECT_MEMBERS = 128;
    public static final int MAX_ARRAY_ELEMENTS = 1_024;
    public static final int MAX_STRING_BYTES = 4_096;
    public static final long MAX_U63 = Long.MAX_VALUE;

    public static final int ED25519_PUBLIC_KEY_BYTES = 32;
    public static final int ED25519_SIGNATURE_BYTES = 64;

    public static final int VERDICT_INVALID = 0;
    public static final int VERDICT_VALID = 1;

    public static final int REJECT_SYNTAX = 1;
    public static final int REJECT_BUDGET = 2;
    public static final int REJECT_ENVELOPE = 3;
    public static final int REJECT_COMMON_PAYLOAD = 4;
    public static final int REJECT_HEX = 5;
    public static final int REJECT_BYTE_TUPLE = 6;

    public static final String ALGORITHM = "Ed25519";
    public static final long SCHEMA = 1L;
    public static final long VERSION = 1L;

    public static final String ROOT_REGISTRY = "root-registry";
    public static final String REVOCATION_SNAPSHOT = "revocation-snapshot";
    public static final String SOURCE_SNAPSHOT_DECLARATION = "source-snapshot-declaration";
    public static final String ANNOTATION_SOURCE_DECLARATION = "annotation-source-declaration";

    private static final Set<String> ENVELOPE_FIELDS = Set.of("signature_hex", "signed_payload");
    private static final Set<String> COMMON_PAYLOAD_FIELDS = Set.of(
            "algorithm", "key_id", "kind", "schema", "version");
    private static final Map<String, Bytes> DOMAIN_PREFIXES = Map.of(
            ROOT_REGISTRY, Bytes.fromAsciiLiteral("PIDSLCA-G0/root-registry/v1\u0000"),
            REVOCATION_SNAPSHOT, Bytes.fromAsciiLiteral("PIDSLCA-G0/revocation-snapshot/v1\u0000"),
            SOURCE_SNAPSHOT_DECLARATION,
            Bytes.fromAsciiLiteral("PIDSLCA-G0/source-snapshot-declaration/v1\u0000"),
            ANNOTATION_SOURCE_DECLARATION,
            Bytes.fromAsciiLiteral("PIDSLCA-G0/annotation-source-declaration/v1\u0000"));

    private GovCjson() {
    }

    /** 所有 core 输入的有限 JSON 值；宿主对象布局不是协议语义。 */
    public sealed interface Value permits TextValue, UIntValue, ArrayValue, ObjectValue {
    }

    /** ASCII text 的宿主视图；编码或 schema 边界会再次执行协议校验。 */
    public record TextValue(String value) implements Value {
        public TextValue {
            Objects.requireNonNull(value, "value");
        }
    }

    /** 非负 u63 的宿主视图；负值会在 core 边界 fail closed。 */
    public record UIntValue(long value) implements Value {
    }

    /** 保序 JSON array 的宿主视图。 */
    public record ArrayValue(List<Value> values) implements Value {
        public ArrayValue {
            values = List.copyOf(values);
        }
    }

    /** JSON object 的宿主视图；encoder 以 ASCII 字节序重新排序。 */
    public static final class ObjectValue implements Value {
        private final Map<String, Value> values;

        public ObjectValue(Map<String, ? extends Value> values) {
            Objects.requireNonNull(values, "values");
            LinkedHashMap<String, Value> copied = new LinkedHashMap<>();
            for (Map.Entry<String, ? extends Value> entry : values.entrySet()) {
                copied.put(Objects.requireNonNull(entry.getKey(), "object key"),
                        Objects.requireNonNull(entry.getValue(), "object value"));
            }
            this.values = Collections.unmodifiableMap(copied);
        }

        public Map<String, Value> values() {
            return values;
        }

        public Value value(String field) {
            return values.get(field);
        }
    }

    /** 不可变有限 u8 序列，所有访问都显式转为 unsigned int。 */
    public static final class Bytes {
        private static final char[] HEX = "0123456789abcdef".toCharArray();
        private final byte[] data;

        private Bytes(byte[] data, boolean copy) {
            this.data = copy ? Arrays.copyOf(data, data.length) : data;
        }

        public static Bytes copyOf(byte[] data) {
            return new Bytes(Objects.requireNonNull(data, "data"), true);
        }

        public static Bytes fromAsciiLiteral(String value) {
            ByteBuilder builder = new ByteBuilder();
            for (int index = 0; index < value.length(); index++) {
                char character = value.charAt(index);
                if (character > 0x7f) {
                    throw new IllegalArgumentException("ASCII literal contains a non-ASCII character");
                }
                builder.appendUnsigned(character);
            }
            return builder.toBytes();
        }

        public static Bytes fromLowerHex(String value) {
            return requireLowerHex(value, "hex", -1);
        }

        /**
         * 验证跨语言 adapter 传入的显式 unsigned 整数数组。
         *
         * <p>Java {@code byte[]} 自身无法表达大于 255 的候选值；需要验证该失败面时，adapter
         * 应调用本方法，而不是先窄化转换。</p>
         */
        public static Bytes fromUnsignedIntArray(int[] values, int expectedLength) {
            if (expectedLength < 0) {
                throw new IllegalArgumentException("expected byte length must be non-negative");
            }
            if (values == null || values.length != expectedLength) {
                fail(REJECT_BYTE_TUPLE, "unsigned byte array has the wrong fixed length");
            }
            byte[] result = new byte[expectedLength];
            for (int index = 0; index < expectedLength; index++) {
                int item = values[index];
                if (item < 0 || item > 255) {
                    fail(REJECT_BYTE_TUPLE, "unsigned byte array contains a value outside 0..255");
                }
                result[index] = (byte) item;
            }
            return new Bytes(result, false);
        }

        public int length() {
            return data.length;
        }

        public int unsignedAt(int index) {
            return Byte.toUnsignedInt(data[index]);
        }

        public byte[] toByteArray() {
            return Arrays.copyOf(data, data.length);
        }

        public String toLowerHex() {
            char[] result = new char[data.length * 2];
            for (int index = 0; index < data.length; index++) {
                int value = Byte.toUnsignedInt(data[index]);
                result[index * 2] = HEX[value >>> 4];
                result[index * 2 + 1] = HEX[value & 0x0f];
            }
            return new String(result);
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof Bytes bytes && Arrays.equals(data, bytes.data);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(data);
        }
    }

    /** 固定整数拒绝码；异常文字不属于跨语言协议值。 */
    public static final class GovException extends RuntimeException {
        private static final long serialVersionUID = 1L;
        private final int code;

        public GovException(int code, String message) {
            super(message);
            this.code = code;
        }

        public int code() {
            return code;
        }
    }

    /** 仅在公共字段通过后构造的 detached envelope，仍完全未验签。 */
    public static final class WireEnvelope {
        private final String kind;
        private final long schema;
        private final long version;
        private final String keyId;
        private final Bytes canonicalSignedPayload;
        private final Bytes detachedSignature;
        private final Bytes domainPrefix;
        private final Bytes message;
        private final Bytes documentIdentity;

        private WireEnvelope(
                String kind,
                long schema,
                long version,
                String keyId,
                Bytes canonicalSignedPayload,
                Bytes detachedSignature,
                Bytes domainPrefix,
                Bytes message,
                Bytes documentIdentity) {
            this.kind = kind;
            this.schema = schema;
            this.version = version;
            this.keyId = keyId;
            this.canonicalSignedPayload = canonicalSignedPayload;
            this.detachedSignature = detachedSignature;
            this.domainPrefix = domainPrefix;
            this.message = message;
            this.documentIdentity = documentIdentity;
        }

        public String kind() {
            return kind;
        }

        public long schema() {
            return schema;
        }

        public long version() {
            return version;
        }

        public String keyId() {
            return keyId;
        }

        public Bytes canonicalSignedPayload() {
            return canonicalSignedPayload;
        }

        public Bytes detachedSignature() {
            return detachedSignature;
        }

        public Bytes domainPrefix() {
            return domainPrefix;
        }

        public Bytes message() {
            return message;
        }

        public Bytes documentIdentity() {
            return documentIdentity;
        }

        public String status() {
            return STATUS_REFERENCE_ONLY;
        }
    }

    /** 以唯一 GOV-CJSON-1 bytes 编码 object root。 */
    public static Bytes encode(ObjectValue value) {
        BoundedByteWriter writer = new BoundedByteWriter(MAX_DOCUMENT_BYTES);
        encodeValue(value, 1, "GOV-CJSON-1 root", writer);
        return writer.toBytes();
    }

    /** 解析并重编码核对 GOV-CJSON-1 object root，拒绝任意宽松物理表示。 */
    public static ObjectValue parse(Bytes payload) {
        ObjectValue value = new Parser(payload).parseRoot();
        if (!encode(value).equals(payload)) {
            fail(REJECT_SYNTAX, "GOV-CJSON-1 payload is not the unique canonical encoding");
        }
        return value;
    }

    /** 以固定 64-byte detached signature 构造唯一 physical envelope。 */
    public static Bytes encodeEnvelope(ObjectValue signedPayload, Bytes detachedSignature) {
        validateCommonSignedPayload(signedPayload);
        requireByteLength(detachedSignature, ED25519_SIGNATURE_BYTES, "detached signature");
        LinkedHashMap<String, Value> envelope = new LinkedHashMap<>();
        envelope.put("signature_hex", new TextValue(detachedSignature.toLowerHex()));
        envelope.put("signed_payload", signedPayload);
        Bytes encoded = encode(new ObjectValue(envelope));
        parseEnvelope(encoded);
        return encoded;
    }

    /** 解析 envelope，构造固定 domain message 与 SHA-256 identity；绝不验签。 */
    public static WireEnvelope parseEnvelope(Bytes payload) {
        ObjectValue envelope = parse(payload);
        requireExactFields(envelope, ENVELOPE_FIELDS, REJECT_ENVELOPE, "envelope");
        Value rawSignature = envelope.value("signature_hex");
        if (!(rawSignature instanceof TextValue signatureText)) {
            fail(REJECT_HEX, "signature_hex is not fixed lowercase hex");
            throw new AssertionError("unreachable");
        }
        Bytes signature = requireLowerHex(
                signatureText.value(),
                "signature_hex", ED25519_SIGNATURE_BYTES);
        Value rawPayload = envelope.value("signed_payload");
        if (!(rawPayload instanceof ObjectValue signedPayload)) {
            fail(REJECT_COMMON_PAYLOAD, "signed_payload must be an object");
            throw new AssertionError("unreachable");
        }
        CommonPayload common = validateCommonSignedPayload(signedPayload);
        Bytes canonicalSignedPayload = encode(signedPayload);
        Bytes domainPrefix = domainPrefix(common.kind());
        Bytes message = concat(domainPrefix, canonicalSignedPayload);
        return new WireEnvelope(
                common.kind(), common.schema(), common.version(), common.keyId(),
                canonicalSignedPayload, signature, domainPrefix, message, sha256(message));
    }

    /** 返回已注册 kind 的固定 ASCII domain prefix，拒绝动态 domain。 */
    public static Bytes domainPrefix(String kind) {
        requireToken(kind, "domain kind");
        Bytes result = DOMAIN_PREFIXES.get(kind);
        if (result == null) {
            fail(REJECT_COMMON_PAYLOAD, "domain kind is not registered");
        }
        return result;
    }

    /** SHA-256 是固定 domain message 的 identity primitive，不是 trust verdict。 */
    public static Bytes sha256(Bytes value) {
        try {
            return new Bytes(MessageDigest.getInstance("SHA-256").digest(value.data), false);
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("Java 17 SHA-256 is unavailable", exception);
        }
    }

    /** 用于固定 message 拼接的有限字节操作。 */
    public static Bytes concat(Bytes left, Bytes right) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        if (left.length() > Integer.MAX_VALUE - right.length()) {
            fail(REJECT_BUDGET, "byte concatenation exceeds implementation bound");
        }
        byte[] combined = new byte[left.length() + right.length()];
        System.arraycopy(left.data, 0, combined, 0, left.length());
        System.arraycopy(right.data, 0, combined, left.length(), right.length());
        return new Bytes(combined, false);
    }

    static String requireText(Value value, String label, int code) {
        if (!(value instanceof TextValue text)) {
            fail(code, label + " must be an ASCII string");
            throw new AssertionError("unreachable");
        }
        return requireAsciiText(text.value(), label, code);
    }

    static String requireAsciiText(String value, String label, int code) {
        if (value == null) {
            fail(code, label + " must be an ASCII string");
        }
        for (int index = 0; index < value.length(); index++) {
            char character = value.charAt(index);
            if (character > 0x7e) {
                fail(REJECT_SYNTAX, label + " contains a non-ASCII character");
            }
            if (character < 0x20) {
                fail(REJECT_BUDGET, label + " contains a non-printable ASCII byte");
            }
        }
        if (value.length() > MAX_STRING_BYTES) {
            fail(REJECT_BUDGET, label + " exceeds ASCII string budget");
        }
        return value;
    }

    static long requireU63(Value value, String label, int code) {
        if (!(value instanceof UIntValue integer) || integer.value() < 0) {
            fail(code, label + " must be a u63 integer");
            throw new AssertionError("unreachable");
        }
        return integer.value();
    }

    static long requireU63(long value, String label, int code) {
        if (value < 0) {
            fail(code, label + " must be a u63 integer");
        }
        return value;
    }

    static void requireExactFields(ObjectValue value, Set<String> fields, int code, String label) {
        if (!value.values().keySet().equals(fields)) {
            fail(code, label + " field set is not exact");
        }
    }

    static int compareAscii(String left, String right) {
        int common = Math.min(left.length(), right.length());
        for (int index = 0; index < common; index++) {
            int difference = left.charAt(index) - right.charAt(index);
            if (difference != 0) {
                return difference;
            }
        }
        return Integer.compare(left.length(), right.length());
    }

    /** 以原始 u8 字节词典序排序 host collection witness，绝不使用 Java signed byte 顺序。 */
    static int compareUnsignedBytes(Bytes left, Bytes right) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        int common = Math.min(left.length(), right.length());
        for (int index = 0; index < common; index++) {
            int difference = left.unsignedAt(index) - right.unsignedAt(index);
            if (difference != 0) {
                return difference;
            }
        }
        return Integer.compare(left.length(), right.length());
    }

    private static CommonPayload validateCommonSignedPayload(ObjectValue value) {
        if (!value.values().keySet().containsAll(COMMON_PAYLOAD_FIELDS)) {
            fail(REJECT_COMMON_PAYLOAD, "signed_payload lacks common governance fields");
        }
        String algorithm = requireToken(
                requireText(value.value("algorithm"), "algorithm", REJECT_SYNTAX), "algorithm");
        if (!ALGORITHM.equals(algorithm)) {
            fail(REJECT_COMMON_PAYLOAD, "algorithm is not fixed to Ed25519");
        }
        String kind = requireToken(
                requireText(value.value("kind"), "kind", REJECT_SYNTAX), "kind");
        if (!DOMAIN_PREFIXES.containsKey(kind)) {
            fail(REJECT_COMMON_PAYLOAD, "kind is not registered");
        }
        long schema = requireU63(value.value("schema"), "schema", REJECT_SYNTAX);
        long version = requireU63(value.value("version"), "version", REJECT_SYNTAX);
        if (schema != SCHEMA || version != VERSION) {
            fail(REJECT_COMMON_PAYLOAD, "schema/version is not registered");
        }
        String keyId = requireToken(
                requireText(value.value("key_id"), "key_id", REJECT_SYNTAX), "key_id");
        return new CommonPayload(kind, schema, version, keyId);
    }

    private static String requireToken(String value, String label) {
        requireAsciiText(value, label, REJECT_SYNTAX);
        if (value.isEmpty() || value.length() > 128) {
            fail(REJECT_COMMON_PAYLOAD, label + " is not a controlled ASCII token");
        }
        for (int index = 0; index < value.length(); index++) {
            char character = value.charAt(index);
            if (!isTokenCharacter(character)) {
                fail(REJECT_COMMON_PAYLOAD, label + " is not a controlled ASCII token");
            }
        }
        return value;
    }

    private static boolean isTokenCharacter(char character) {
        return (character >= 'A' && character <= 'Z')
                || (character >= 'a' && character <= 'z')
                || (character >= '0' && character <= '9')
                || character == '.' || character == '_' || character == ':' || character == '-';
    }

    private static String requireFieldName(String value, String label) {
        requireAsciiText(value, label, REJECT_SYNTAX);
        if (value.isEmpty() || !isAsciiLetter(value.charAt(0))) {
            fail(REJECT_SYNTAX, label + " is not a GOV-CJSON field name");
        }
        for (int index = 0; index < value.length(); index++) {
            if (!isTokenCharacter(value.charAt(index))) {
                fail(REJECT_SYNTAX, label + " is not a GOV-CJSON field name");
            }
        }
        return value;
    }

    private static boolean isAsciiLetter(char character) {
        return (character >= 'A' && character <= 'Z') || (character >= 'a' && character <= 'z');
    }

    private static Bytes requireLowerHex(String value, String label, int byteCount) {
        requireAsciiText(value, label, REJECT_SYNTAX);
        if ((byteCount >= 0 && value.length() != byteCount * 2)
                || (byteCount < 0 && (value.length() & 1) != 0)) {
            fail(REJECT_HEX, label + " is not fixed lowercase hex");
        }
        ByteBuilder result = new ByteBuilder();
        for (int index = 0; index < value.length(); index += 2) {
            int high = hexValue(value.charAt(index));
            int low = hexValue(value.charAt(index + 1));
            if (high < 0 || low < 0) {
                fail(REJECT_HEX, label + " is not fixed lowercase hex");
            }
            result.appendUnsigned((high << 4) | low);
        }
        return result.toBytes();
    }

    private static int hexValue(char character) {
        return character >= '0' && character <= '9'
                ? character - '0'
                : character >= 'a' && character <= 'f' ? character - 'a' + 10 : -1;
    }

    private static void requireByteLength(Bytes value, int expected, String label) {
        if (value == null || value.length() != expected) {
            fail(REJECT_BYTE_TUPLE, label + " is not a fixed-length byte sequence");
        }
    }

    private static void encodeValue(Value value, int depth, String label, BoundedByteWriter writer) {
        if (depth > MAX_DEPTH) {
            fail(REJECT_BUDGET, label + " exceeds GOV-CJSON maximum depth");
        }
        if (value instanceof TextValue text) {
            encodeString(text.value(), label, writer);
            return;
        }
        if (value instanceof UIntValue integer) {
            long number = requireU63(integer.value(), label, REJECT_SYNTAX);
            writer.appendAscii(Long.toString(number));
            return;
        }
        if (value instanceof ArrayValue array) {
            if (array.values().size() > MAX_ARRAY_ELEMENTS) {
                fail(REJECT_BUDGET, label + " array exceeds element budget");
            }
            writer.appendUnsigned('[');
            for (int index = 0; index < array.values().size(); index++) {
                if (index > 0) {
                    writer.appendUnsigned(',');
                }
                encodeValue(array.values().get(index), depth + 1, label + "[]", writer);
            }
            writer.appendUnsigned(']');
            return;
        }
        if (value instanceof ObjectValue object) {
            if (object.values().size() > MAX_OBJECT_MEMBERS) {
                fail(REJECT_BUDGET, label + " object exceeds member budget");
            }
            List<Map.Entry<String, Value>> entries = new ArrayList<>(object.values().entrySet());
            for (Map.Entry<String, Value> entry : entries) {
                requireFieldName(entry.getKey(), label + " field");
            }
            entries.sort(Comparator.comparing(Map.Entry::getKey, GovCjson::compareAscii));
            writer.appendUnsigned('{');
            for (int index = 0; index < entries.size(); index++) {
                if (index > 0) {
                    writer.appendUnsigned(',');
                }
                Map.Entry<String, Value> entry = entries.get(index);
                encodeString(entry.getKey(), label + " field", writer);
                writer.appendUnsigned(':');
                encodeValue(entry.getValue(), depth + 1, label + "." + entry.getKey(), writer);
            }
            writer.appendUnsigned('}');
            return;
        }
        fail(REJECT_SYNTAX, label + " contains an unsupported GOV-CJSON type");
    }

    private static void encodeString(String value, String label, BoundedByteWriter writer) {
        String text = requireAsciiText(value, label, REJECT_SYNTAX);
        writer.appendUnsigned('"');
        int representationLength = 0;
        for (int index = 0; index < text.length(); index++) {
            char character = text.charAt(index);
            int addition = character == '"' || character == '\\' ? 2 : 1;
            if (addition > MAX_STRING_BYTES - representationLength) {
                fail(REJECT_BUDGET, label + " representation exceeds string byte budget");
            }
            if (addition == 2) {
                writer.appendUnsigned('\\');
            }
            writer.appendUnsigned(character);
            representationLength += addition;
        }
        writer.appendUnsigned('"');
    }

    private static void fail(int code, String message) {
        throw new GovException(code, message);
    }

    private record CommonPayload(String kind, long schema, long version, String keyId) {
    }

    /** GOV-CJSON-1 的小型递归下降 parser，cursor 是唯一可变状态。 */
    private static final class Parser {
        private final byte[] payload;
        private int cursor;

        private Parser(Bytes source) {
            if (source == null || source.length() == 0) {
                fail(REJECT_SYNTAX, "GOV-CJSON-1 payload must be non-empty bytes");
            }
            if (source.length() > MAX_DOCUMENT_BYTES) {
                fail(REJECT_BUDGET, "GOV-CJSON-1 document exceeds byte budget");
            }
            this.payload = source.data;
            if (startsWithBom(payload)) {
                fail(REJECT_SYNTAX, "GOV-CJSON-1 forbids BOM");
            }
            for (byte item : payload) {
                if (Byte.toUnsignedInt(item) >= 0x80) {
                    fail(REJECT_SYNTAX, "GOV-CJSON-1 forbids non-ASCII bytes");
                }
            }
        }

        private ObjectValue parseRoot() {
            Value value = parseValue(1);
            if (!(value instanceof ObjectValue object)) {
                fail(REJECT_SYNTAX, "GOV-CJSON-1 root must be an object");
                throw new AssertionError("unreachable");
            }
            if (cursor != payload.length) {
                fail(REJECT_SYNTAX, "GOV-CJSON-1 contains trailing bytes");
            }
            return object;
        }

        private Value parseValue(int depth) {
            if (depth > MAX_DEPTH) {
                fail(REJECT_BUDGET, "GOV-CJSON-1 exceeds maximum depth");
            }
            int current = peek();
            if (current == '{') {
                return parseObject(depth);
            }
            if (current == '[') {
                return parseArray(depth);
            }
            if (current == '"') {
                return parseString("GOV-CJSON-1 string");
            }
            if (current >= '0' && current <= '9') {
                return new UIntValue(parseU63());
            }
            fail(REJECT_SYNTAX, "GOV-CJSON-1 only allows object/array/string/u63 values");
            throw new AssertionError("unreachable");
        }

        private ObjectValue parseObject(int depth) {
            expect('{', "object open");
            LinkedHashMap<String, Value> result = new LinkedHashMap<>();
            String previous = null;
            if (consumeIf('}')) {
                return new ObjectValue(result);
            }
            while (true) {
                String key = requireFieldName(
                        parseString("GOV-CJSON-1 object key").value(),
                        "GOV-CJSON-1 object key");
                if (previous != null && compareAscii(key, previous) <= 0) {
                    fail(REJECT_SYNTAX, "GOV-CJSON-1 object keys are not strictly ASCII ascending");
                }
                if (result.containsKey(key)) {
                    fail(REJECT_SYNTAX, "GOV-CJSON-1 object contains a duplicate key");
                }
                previous = key;
                if (result.size() >= MAX_OBJECT_MEMBERS) {
                    fail(REJECT_BUDGET, "GOV-CJSON-1 object exceeds member budget");
                }
                expect(':', "object colon");
                result.put(key, parseValue(depth + 1));
                if (consumeIf('}')) {
                    return new ObjectValue(result);
                }
                expect(',', "object comma");
            }
        }

        private ArrayValue parseArray(int depth) {
            expect('[', "array open");
            ArrayList<Value> result = new ArrayList<>();
            if (consumeIf(']')) {
                return new ArrayValue(result);
            }
            while (true) {
                if (result.size() >= MAX_ARRAY_ELEMENTS) {
                    fail(REJECT_BUDGET, "GOV-CJSON-1 array exceeds element budget");
                }
                result.add(parseValue(depth + 1));
                if (consumeIf(']')) {
                    return new ArrayValue(result);
                }
                expect(',', "array comma");
            }
        }

        private TextValue parseString(String label) {
            expect('"', label + " open");
            int contentStart = cursor;
            ByteBuilder result = new ByteBuilder();
            while (true) {
                if (cursor >= payload.length) {
                    fail(REJECT_SYNTAX, label + " is truncated");
                }
                int current = Byte.toUnsignedInt(payload[cursor++]);
                if (current == '"') {
                    if (cursor - contentStart - 1 > MAX_STRING_BYTES) {
                        fail(REJECT_BUDGET, label + " representation exceeds string byte budget");
                    }
                    return new TextValue(result.toAsciiString());
                }
                if (current == '\\') {
                    if (cursor >= payload.length) {
                        fail(REJECT_SYNTAX, label + " escape is truncated");
                    }
                    int escaped = Byte.toUnsignedInt(payload[cursor++]);
                    if (escaped != '"' && escaped != '\\') {
                        fail(REJECT_SYNTAX, label + " contains an unregistered escape");
                    }
                    result.appendUnsigned(escaped);
                } else if (current >= 0x20 && current <= 0x7e && current != '"' && current != '\\') {
                    result.appendUnsigned(current);
                } else {
                    fail(REJECT_SYNTAX, label + " contains an invalid ASCII byte");
                }
                if (result.length() > MAX_STRING_BYTES) {
                    fail(REJECT_BUDGET, label + " exceeds string byte budget");
                }
            }
        }

        private long parseU63() {
            int first = Byte.toUnsignedInt(payload[cursor++]);
            if (first == '0') {
                if (cursor < payload.length) {
                    int next = Byte.toUnsignedInt(payload[cursor]);
                    if (next >= '0' && next <= '9') {
                        fail(REJECT_SYNTAX, "GOV-CJSON-1 integer has a leading zero");
                    }
                }
                return 0L;
            }
            long result = first - '0';
            while (cursor < payload.length) {
                int current = Byte.toUnsignedInt(payload[cursor]);
                if (current < '0' || current > '9') {
                    break;
                }
                int digit = current - '0';
                if (result > (MAX_U63 - digit) / 10L) {
                    fail(REJECT_SYNTAX, "GOV-CJSON-1 integer exceeds u63");
                }
                result = result * 10L + digit;
                cursor++;
            }
            return result;
        }

        private int peek() {
            if (cursor >= payload.length) {
                fail(REJECT_SYNTAX, "GOV-CJSON-1 payload is truncated");
            }
            return Byte.toUnsignedInt(payload[cursor]);
        }

        private void expect(int expected, String label) {
            if (peek() != expected) {
                fail(REJECT_SYNTAX, "GOV-CJSON-1 lacks " + label);
            }
            cursor++;
        }

        private boolean consumeIf(int expected) {
            if (cursor < payload.length && Byte.toUnsignedInt(payload[cursor]) == expected) {
                cursor++;
                return true;
            }
            return false;
        }

        private static boolean startsWithBom(byte[] value) {
            return value.length >= 3
                    && Byte.toUnsignedInt(value[0]) == 0xef
                    && Byte.toUnsignedInt(value[1]) == 0xbb
                    && Byte.toUnsignedInt(value[2]) == 0xbf;
        }
    }

    /**
     * 整个 GOV-CJSON document 共享的有界 writer。
     *
     * <p>每次 append 都在分配或复制前检查 65,536-byte envelope budget；递归 encoder 因而
     * 不会先物化一个超限子树再在 root 处失败。</p>
     */
    private static final class BoundedByteWriter {
        private final int limit;
        private byte[] data = new byte[64];
        private int length;

        private BoundedByteWriter(int limit) {
            if (limit < 0) {
                throw new IllegalArgumentException("byte writer limit must be non-negative");
            }
            this.limit = limit;
        }

        private void appendUnsigned(int value) {
            if (value < 0 || value > 255) {
                throw new IllegalArgumentException("unsigned byte is outside 0..255");
            }
            reserve(1);
            data[length++] = (byte) value;
        }

        private void appendAscii(String value) {
            for (int index = 0; index < value.length(); index++) {
                char character = value.charAt(index);
                if (character > 0x7f) {
                    throw new IllegalArgumentException("writer received non-ASCII text");
                }
                appendUnsigned(character);
            }
        }

        private Bytes toBytes() {
            return new Bytes(Arrays.copyOf(data, length), false);
        }

        private void reserve(int addition) {
            if (addition < 0 || addition > limit - length) {
                fail(REJECT_BUDGET, "GOV-CJSON-1 document exceeds byte budget");
            }
            int required = length + addition;
            if (required <= data.length) {
                return;
            }
            int next = data.length;
            while (next < required) {
                if (next >= limit || next > Integer.MAX_VALUE / 2) {
                    next = required;
                    break;
                }
                next = Math.min(limit, next * 2);
            }
            data = Arrays.copyOf(data, next);
        }
    }

    /** 无 I/O 的有限 byte builder，避免把 Java signed byte 当作协议整数。 */
    private static final class ByteBuilder {
        private byte[] data = new byte[64];
        private int length;

        private void appendUnsigned(int value) {
            if (value < 0 || value > 255) {
                throw new IllegalArgumentException("unsigned byte is outside 0..255");
            }
            ensureCapacity(1);
            data[length++] = (byte) value;
        }

        private void append(Bytes value) {
            Objects.requireNonNull(value, "value");
            ensureCapacity(value.length());
            System.arraycopy(value.data, 0, data, length, value.length());
            length += value.length();
        }

        private int length() {
            return length;
        }

        private Bytes toBytes() {
            return new Bytes(Arrays.copyOf(data, length), false);
        }

        private String toAsciiString() {
            char[] result = new char[length];
            for (int index = 0; index < length; index++) {
                result[index] = (char) Byte.toUnsignedInt(data[index]);
            }
            return new String(result);
        }

        private void ensureCapacity(int addition) {
            if (addition > Integer.MAX_VALUE - length) {
                fail(REJECT_BUDGET, "byte builder exceeds implementation bound");
            }
            int required = length + addition;
            if (required <= data.length) {
                return;
            }
            int next = data.length;
            while (next < required) {
                if (next > Integer.MAX_VALUE / 2) {
                    next = required;
                    break;
                }
                next *= 2;
            }
            data = Arrays.copyOf(data, next);
        }
    }
}
