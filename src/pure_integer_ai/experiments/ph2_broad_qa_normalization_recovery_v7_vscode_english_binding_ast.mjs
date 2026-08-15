import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { createRequire } from 'node:module';

function fail(message) {
	process.stderr.write(`${message}\n`);
	process.exit(2);
}

if (process.argv.length !== 4) {
	fail('usage: extractor <typescript-parser-root> <source-root>');
}

const parserRoot = path.resolve(process.argv[2]);
const sourceRoot = path.resolve(process.argv[3]);
const require = createRequire(import.meta.url);
let ts;
try {
	ts = require(path.join(
		parserRoot, 'node_modules', '@typescript', 'typescript6'));
} catch (error) {
	fail(`typescript parser unavailable: ${String(error)}`);
}

let request;
try {
	request = JSON.parse(fs.readFileSync(0, 'utf8'));
} catch (error) {
	fail(`request unreadable: ${String(error)}`);
}
if (!request || !Array.isArray(request.relative_paths)) {
	fail('relative_paths missing');
}

function staticText(node) {
	if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
		return node.text;
	}
	return null;
}

function propertyName(node) {
	if (ts.isIdentifier(node) || ts.isStringLiteral(node)
			|| ts.isNoSubstitutionTemplateLiteral(node)) {
		return node.text;
	}
	return null;
}

function keyText(node) {
	const direct = staticText(node);
	if (direct !== null) {
		return direct;
	}
	if (!ts.isObjectLiteralExpression(node)) {
		return null;
	}
	for (const property of node.properties) {
		if (!ts.isPropertyAssignment(property)
				|| propertyName(property.name) !== 'key') {
			continue;
		}
		return staticText(property.initializer);
	}
	return null;
}

function calleeName(node) {
	if (ts.isIdentifier(node)) {
		return node.text;
	}
	if (ts.isPropertyAccessExpression(node)) {
		return node.name.text;
	}
	return null;
}

const bindings = [];
const diagnostics = [];
const unsupported = [];
let callCount = 0;

for (const relativePath of request.relative_paths) {
	if (typeof relativePath !== 'string' || relativePath.length === 0
			|| relativePath.includes('\\') || path.posix.isAbsolute(relativePath)
			|| relativePath.split('/').includes('..')) {
		fail('invalid relative source path');
	}
	const absolutePath = path.resolve(sourceRoot, ...relativePath.split('/'));
	if (absolutePath !== sourceRoot
			&& !absolutePath.startsWith(sourceRoot + path.sep)) {
		fail('source path escaped root');
	}
	let sourceText;
	try {
		sourceText = fs.readFileSync(absolutePath, 'utf8');
	} catch (error) {
		fail(`source unreadable: ${relativePath}: ${String(error)}`);
	}
	const kind = relativePath.endsWith('.tsx')
		? ts.ScriptKind.TSX : ts.ScriptKind.TS;
	const sourceFile = ts.createSourceFile(
		relativePath, sourceText, ts.ScriptTarget.Latest, true, kind);
	for (const diagnostic of sourceFile.parseDiagnostics ?? []) {
		diagnostics.push({
			code: diagnostic.code,
			position: diagnostic.start ?? -1,
			relative_path: relativePath,
		});
	}
	const modulePath = relativePath.slice(4).replace(/\.(?:ts|tsx)$/u, '');
	function visit(node) {
		if (ts.isCallExpression(node)) {
			const name = calleeName(node.expression);
			if (name === 'localize' || name === 'localize2') {
				callCount += 1;
				const key = node.arguments.length >= 1
					? keyText(node.arguments[0]) : null;
				const message = node.arguments.length >= 2
					? staticText(node.arguments[1]) : null;
				if (key === null || key.length === 0
						|| message === null || message.length === 0) {
					unsupported.push({
						position: node.getStart(sourceFile),
						reason: key === null
							? 'DYNAMIC_OR_MISSING_KEY'
							: 'DYNAMIC_OR_MISSING_MESSAGE',
						relative_path: relativePath,
					});
				} else {
					bindings.push({
						callee: name,
						key,
						message,
						module: modulePath,
						position: node.getStart(sourceFile),
						relative_path: relativePath,
					});
				}
			}
		}
		ts.forEachChild(node, visit);
	}
	visit(sourceFile);
}

bindings.sort((left, right) =>
	left.module.localeCompare(right.module)
	|| left.key.localeCompare(right.key)
	|| left.relative_path.localeCompare(right.relative_path)
	|| left.position - right.position
	|| left.message.localeCompare(right.message));
diagnostics.sort((left, right) =>
	left.relative_path.localeCompare(right.relative_path)
	|| left.position - right.position || left.code - right.code);
unsupported.sort((left, right) =>
	left.relative_path.localeCompare(right.relative_path)
	|| left.position - right.position || left.reason.localeCompare(right.reason));
const bindingKeys = new Set(bindings.map(item => `${item.module}\0${item.key}`));

process.stdout.write(JSON.stringify({
	binding_key_count: bindingKeys.size,
	bindings,
	call_count: callCount,
	file_count: request.relative_paths.length,
	parse_diagnostics: diagnostics,
	typescript_version: ts.version,
	unsupported,
}));
