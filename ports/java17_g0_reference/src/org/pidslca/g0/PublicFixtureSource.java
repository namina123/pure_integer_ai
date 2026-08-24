package org.pidslca.g0;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * 公开 corpus 的测试 I/O adapter。
 *
 * <p>它只接受 runner 显式传入的 fixture root，并且只解析经检查的相对名称。路径、文件系统、
 * UTF-8 与 SHA readback 都止步于这个类，不能进入 portable core。</p>
 */
final class PublicFixtureSource {
    private final Path root;

    PublicFixtureSource(Path root) {
        if (root == null) {
            throw new IllegalArgumentException("fixture root is required");
        }
        this.root = root.toAbsolutePath().normalize();
    }

    Object loadJson(String relativeName) {
        Path resolved = resolve(relativeName);
        try {
            return FixtureJson.parse(Files.readString(resolved, StandardCharsets.UTF_8));
        } catch (IOException exception) {
            throw new IllegalStateException("cannot read public fixture " + relativeName, exception);
        }
    }

    GovCjson.Bytes loadBytes(String relativeName) {
        Path resolved = resolve(relativeName);
        try {
            return GovCjson.Bytes.copyOf(Files.readAllBytes(resolved));
        } catch (IOException exception) {
            throw new IllegalStateException("cannot read public fixture " + relativeName, exception);
        }
    }

    private Path resolve(String relativeName) {
        if (relativeName == null || relativeName.isEmpty() || relativeName.indexOf('\\') >= 0
                || relativeName.startsWith("/") || relativeName.startsWith(".")) {
            throw new IllegalArgumentException("fixture name is not a vetted relative path");
        }
        String[] segments = relativeName.split("/", -1);
        for (String segment : segments) {
            if (segment.isEmpty() || ".".equals(segment) || "..".equals(segment)) {
                throw new IllegalArgumentException("fixture name contains an invalid path segment");
            }
            for (int index = 0; index < segment.length(); index++) {
                char character = segment.charAt(index);
                if (!((character >= 'a' && character <= 'z')
                        || (character >= 'A' && character <= 'Z')
                        || (character >= '0' && character <= '9')
                        || character == '.' || character == '_' || character == '-')) {
                    throw new IllegalArgumentException("fixture name contains an invalid character");
                }
            }
        }
        Path resolved = root.resolve(relativeName).normalize();
        if (!resolved.startsWith(root)) {
            throw new IllegalArgumentException("fixture path escapes the supplied root");
        }
        return resolved;
    }
}
