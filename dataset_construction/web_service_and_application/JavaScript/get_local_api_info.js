const fs = require('fs');
const path = require('path');
const acorn = require('acorn');
const walk = require('acorn-walk');
const jsdoc = require('jsdoc-api');

// 替换为要提取的库
const packageName = "express";
const modulePath = path.dirname(require.resolve(packageName));
const libPath = path.join(modulePath, 'lib');

// 递归读取目录中的所有 JS 文件
function readFilesRecursively(dir) {
    const results = [];
    
    try {
        const files = fs.readdirSync(dir);
        
        for (const file of files) {
            const fullPath = path.join(dir, file);
            
            try {
                const stat = fs.statSync(fullPath);
                
                if (stat.isDirectory()) {
                    results.push(...readFilesRecursively(fullPath));
                } else if (path.extname(fullPath) === '.js') {
                    results.push(fullPath);
                }
            } catch (e) {
                console.error(`Error accessing ${fullPath}:`, e.message);
            }
        }
    } catch (e) {
        console.error(`Error reading directory ${dir}:`, e.message);
    }
    
    return results;
}

// 提取函数签名
function extractSignature(node) {
    if (!node.params) return '()';
    
    return '(' + node.params.map(param => {
        if (!param) return '?';
        
        if (param.type === 'Identifier') {
            return param.name;
        } else if (param.type === 'AssignmentPattern' && param.left) {
            return `${param.left.name} = ${extractDefaultValue(param.right)}`;
        } else if (param.type === 'RestElement' && param.argument) {
            return `...${param.argument.name}`;
        }
        return '?';
    }).join(', ') + ')';
}

// 提取默认值
function extractDefaultValue(node) {
    if (!node) return '?';
    
    if (node.type === 'Literal') return JSON.stringify(node.value);
    if (node.type === 'Identifier') return node.name;
    if (node.type === 'ObjectExpression') return '{}';
    if (node.type === 'ArrayExpression') return '[]';
    if (node.type === 'CallExpression' && node.callee) {
        return `${node.callee.name}()`;
    }
    return '?';
}

// 安全获取函数名称
function getSafeFunctionName(node) {
    try {
        // 函数声明
        if (node.id && node.id.name) return node.id.name;
        
        // 赋值表达式
        if (node.parent && node.parent.type === 'AssignmentExpression') {
            const left = node.parent.left;
            
            // 对象属性赋值 (app.method = function)
            if (left.type === 'MemberExpression' && left.property) {
                return left.property.name || left.property.value;
            }
            
            // 变量赋值 (const method = function)
            if (left.type === 'Identifier') {
                return left.name;
            }
        }
        
        // 对象属性 (methods: { get() {...} })
        if (node.parent && node.parent.type === 'Property' && node.parent.key) {
            return node.parent.key.name || node.parent.key.value;
        }
        
        // 导出语句 (module.exports = function)
        if (node.parent && node.parent.type === 'ExportDefaultDeclaration') {
            return 'default';
        }
    } catch (e) {
        console.error('Error getting function name:', e.message);
    }
    
    return 'anonymous';
}

function formatJSDoc(doc) {
    if (!doc) return null;
    
    let docString = doc.description || '';
    
    // 参数 (@param)
    if (doc.params && doc.params.length > 0) {
        docString += '\n\nParameters:';
        doc.params.forEach(param => {
            const type = param.type ? `{${param.type.names.join('|')}}` : '';
            const name = param.name || '';
            const desc = param.description || '';
            docString += `\n  @param ${type} ${name} ${desc}`;
        });
    }
    
    // 返回值 (@returns)
    if (doc.returns && doc.returns.length > 0) {
        doc.returns.forEach(ret => {
            const type = ret.type ? `{${ret.type.names.join('|')}}` : '';
            const desc = ret.description || '';
            docString += `\n\n@returns ${type} ${desc}`;
        });
    }
    
    // 示例 (@example)
    if (doc.examples && doc.examples.length > 0) {
        docString += '\n\nExamples:';
        doc.examples.forEach((ex, i) => {
            docString += `\n\nExample ${i + 1}:\n${ex}`;
        });
    }
    
    // 其他重要标签
    const otherTags = ['access', 'author', 'copyright', 'deprecated', 'see', 'since', 'version'];
    otherTags.forEach(tag => {
        if (doc[tag]) {
            docString += `\n\n@${tag} ${doc[tag]}`;
        }
    });
    
    return docString.trim();
}

// 主处理函数
async function extractExpressAPI() {
    const apiList = {};
    const jsFiles = readFilesRecursively(libPath);
    
    console.log(`Found ${jsFiles.length} JavaScript files in ${libPath}`);
    
    for (const filePath of jsFiles) {
        try {
            console.log(`Processing: ${filePath}`);
            const code = fs.readFileSync(filePath, 'utf-8');
            
            // 使用 acorn 解析代码
            const ast = acorn.parse(code, {
                ecmaVersion: 'latest',
                sourceType: 'script',
                locations: true,
                allowReturnOutsideFunction: true
            });
            
            // 存储本文件的所有函数
            const fileFunctions = {};
            
            // 遍历 AST
            walk.simple(ast, {
                // 处理函数声明
                FunctionDeclaration(node) {
                    const functionName = getSafeFunctionName(node);
                    if (functionName === 'anonymous') return;
                    
                    const signature = extractSignature(node);
                    const sourceCode = code.substring(node.start, node.end);
                    
                    fileFunctions[functionName] = {
                        name: functionName,
                        signature,
                        sourceCode,
                        filePath
                    };
                },
                
                // 处理函数表达式
                FunctionExpression(node) {
                    const functionName = getSafeFunctionName(node);
                    if (functionName === 'anonymous') return;
                    
                    const signature = extractSignature(node);
                    const sourceCode = code.substring(node.start, node.end);
                    
                    fileFunctions[functionName] = {
                        name: functionName,
                        signature,
                        sourceCode,
                        filePath
                    };
                },
                
                // 处理箭头函数
                ArrowFunctionExpression(node) {
                    const functionName = getSafeFunctionName(node);
                    if (functionName === 'anonymous') return;
                    
                    const signature = extractSignature(node);
                    const sourceCode = code.substring(node.start, node.end);
                    
                    fileFunctions[functionName] = {
                        name: functionName,
                        signature,
                        sourceCode,
                        filePath
                    };
                }
            });
            
            // 提取本文件的 JSDoc 文档
            let fileDocs = [];
            try {
                fileDocs = await jsdoc.explain({ files: filePath });
                console.log(`  Extracted ${fileDocs.length} JSDoc entries`);
            } catch (e) {
                console.error(`  Error extracting JSDoc for ${filePath}:`, e.message);
            }
            
            // 将函数添加到全局 API 列表
            for (const [funcName, funcData] of Object.entries(fileFunctions)) {
                // 查找匹配的文档
                let doc = fileDocs.find(d => 
                    d.kind === 'function' &&
                    d.name === funcName
                );
                
                const fileName = path.basename(filePath, '.js');
                const apiKey = `${fileName}.${funcName}`
                apiList[apiKey] = {
                    package: packageName,
                    name: funcName,
                    signature: funcData.signature,
                    docstring: formatJSDoc(doc),
                    source_code: funcData.sourceCode,
                    // file: path.relative(modulePath, funcData.filePath)
                };
            }
            
        } catch (e) {
            console.error(`Error processing ${filePath}:`, e.message);
            if (e.loc) {
                console.error(`  Error at line ${e.loc.line}, column ${e.loc.column}`);
            }
        }
    }
    
    // 写入 JSON 文件
    const outputPath = `${packageName}_raw_api.json`;
    fs.writeFileSync(outputPath, JSON.stringify(apiList, null, 2));
    console.log(`\n✅ Successfully exported ${Object.keys(apiList).length} functions to ${outputPath}`);
}

// 执行提取
extractExpressAPI().catch(console.error);