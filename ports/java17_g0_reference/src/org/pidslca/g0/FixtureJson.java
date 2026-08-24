package org.pidslca.g0;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 公开 conformance fixture 的测试适配 parser。
 *
 * <p>这不是 GOV-CJSON-1 core：它只让 runner 读取格式化的 fixture JSON，并允许 JSON
 * whitespace、boolean、null 与标准 escape。core 仍只由 {@link GovCjson} 的受限 byte parser
 * 和 encoder 定义。</p>
 */
final class FixtureJson {
    private final String source;
    private int cursor;

    private FixtureJson(String source) {
        this.source = source;
    }

    static Object parse(String source) {
        FixtureJson parser = new FixtureJson(source);
        Object result = parser.parseValue();
        parser.skipWhitespace();
        if (parser.cursor != parser.source.length()) {
            throw parser.error("fixture contains trailing text");
        }
        return result;
    }

    private Object parseValue() {
        skipWhitespace();
        if (cursor >= source.length()) {
            throw error("fixture is truncated");
        }
        return switch (source.charAt(cursor)) {
            case '{' -> parseObject();
            case '[' -> parseArray();
            case '"' -> parseString();
            case 't' -> parseLiteral("true", Boolean.TRUE);
            case 'f' -> parseLiteral("false", Boolean.FALSE);
            case 'n' -> parseLiteral("null", null);
            default -> parseInteger();
        };
    }

    private Map<String, Object> parseObject() {
        expect('{');
        skipWhitespace();
        Map<String, Object> result = new LinkedHashMap<>();
        if (consumeIf('}')) {
            return result;
        }
        while (true) {
            skipWhitespace();
            if (peek() != '"') {
                throw error("fixture object key must be a string");
            }
            String key = parseString();
            if (result.containsKey(key)) {
                throw error("fixture object contains a duplicate key");
            }
            skipWhitespace();
            expect(':');
            result.put(key, parseValue());
            skipWhitespace();
            if (consumeIf('}')) {
                return result;
            }
            expect(',');
        }
    }

    private List<Object> parseArray() {
        expect('[');
        skipWhitespace();
        List<Object> result = new ArrayList<>();
        if (consumeIf(']')) {
            return result;
        }
        while (true) {
            result.add(parseValue());
            skipWhitespace();
            if (consumeIf(']')) {
                return result;
            }
            expect(',');
        }
    }

    private String parseString() {
        expect('"');
        StringBuilder result = new StringBuilder();
        while (true) {
            if (cursor >= source.length()) {
                throw error("fixture string is truncated");
            }
            char character = source.charAt(cursor++);
            if (character == '"') {
                return result.toString();
            }
            if (character < 0x20) {
                throw error("fixture string contains a control character");
            }
            if (character != '\\') {
                result.append(character);
                continue;
            }
            if (cursor >= source.length()) {
                throw error("fixture escape is truncated");
            }
            char escaped = source.charAt(cursor++);
            switch (escaped) {
                case '"', '\\', '/' -> result.append(escaped);
                case 'b' -> result.append('\b');
                case 'f' -> result.append('\f');
                case 'n' -> result.append('\n');
                case 'r' -> result.append('\r');
                case 't' -> result.append('\t');
                case 'u' -> result.append((char) parseHexQuad());
                default -> throw error("fixture contains an invalid escape");
            }
        }
    }

    private int parseHexQuad() {
        if (cursor + 4 > source.length()) {
            throw error("fixture unicode escape is truncated");
        }
        int result = 0;
        for (int index = 0; index < 4; index++) {
            int digit = Character.digit(source.charAt(cursor++), 16);
            if (digit < 0) {
                throw error("fixture unicode escape is malformed");
            }
            result = (result << 4) | digit;
        }
        return result;
    }

    private Object parseLiteral(String expected, Object value) {
        if (!source.startsWith(expected, cursor)) {
            throw error("fixture literal is malformed");
        }
        cursor += expected.length();
        return value;
    }

    private Long parseInteger() {
        int start = cursor;
        if (consumeIf('-')) {
            if (cursor >= source.length() || !isDigit(source.charAt(cursor))) {
                throw error("fixture number is malformed");
            }
        }
        if (cursor >= source.length() || !isDigit(source.charAt(cursor))) {
            throw error("fixture value is unsupported");
        }
        if (source.charAt(cursor) == '0') {
            cursor++;
        } else {
            while (cursor < source.length() && isDigit(source.charAt(cursor))) {
                cursor++;
            }
        }
        if (cursor < source.length() && (source.charAt(cursor) == '.'
                || source.charAt(cursor) == 'e' || source.charAt(cursor) == 'E')) {
            throw error("fixture decimal numbers are unsupported");
        }
        try {
            return Long.valueOf(source.substring(start, cursor));
        } catch (NumberFormatException exception) {
            throw error("fixture integer exceeds signed 64-bit range");
        }
    }

    private void skipWhitespace() {
        while (cursor < source.length()) {
            char character = source.charAt(cursor);
            if (character == ' ' || character == '\t' || character == '\r' || character == '\n') {
                cursor++;
            } else {
                return;
            }
        }
    }

    private char peek() {
        if (cursor >= source.length()) {
            throw error("fixture is truncated");
        }
        return source.charAt(cursor);
    }

    private void expect(char expected) {
        if (peek() != expected) {
            throw error("fixture delimiter is missing");
        }
        cursor++;
    }

    private boolean consumeIf(char expected) {
        if (cursor < source.length() && source.charAt(cursor) == expected) {
            cursor++;
            return true;
        }
        return false;
    }

    private static boolean isDigit(char character) {
        return character >= '0' && character <= '9';
    }

    private IllegalArgumentException error(String message) {
        return new IllegalArgumentException(message + " at fixture character " + cursor);
    }
}
