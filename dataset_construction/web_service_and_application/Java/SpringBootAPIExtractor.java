import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.PackageDeclaration;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.comments.Comment;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class SpringBootAPIExtractor {

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.err.println("Usage: java SpringBootAPIExtractor <source-path> [output-file]");
            System.err.println("Example: java SpringBootAPIExtractor /path/to/spring-boot/src/main/java");
            System.exit(1);
        }
        
        String sourcePath = args[0];
        String outputFile = args.length > 1 ? args[1] : "spring_boot_api.json";
        
        System.out.println("Starting extraction from: " + sourcePath);
        Map<String, MethodInfo> apiMap = extractAPIInfo(sourcePath);
        writeToJson(apiMap, outputFile);
        
        System.out.println("✅ Successfully exported " + apiMap.size() + " methods to " + outputFile);
    }

    private static Map<String, MethodInfo> extractAPIInfo(String sourcePath) throws IOException {
        Map<String, MethodInfo> apiMap = new HashMap<>();
        List<Path> javaFiles = findJavaFiles(sourcePath);
        
        if (javaFiles.isEmpty()) {
            System.err.println("⚠️ No Java files found in: " + sourcePath);
            return apiMap;
        }
        
        System.out.println("Found " + javaFiles.size() + " Java files to process");

        for (Path filePath : javaFiles) {
            try (FileInputStream in = new FileInputStream(filePath.toFile())) {
                JavaParser parser = new JavaParser();
                ParseResult<CompilationUnit> result = parser.parse(in);
                
                if (result.isSuccessful() && result.getResult().isPresent()) {
                    CompilationUnit cu = result.getResult().get();
                    
                    String packageName = cu.getPackageDeclaration()
                            .map(PackageDeclaration::getNameAsString)
                            .orElse("default");
                    
                    List<ClassOrInterfaceDeclaration> classes = cu.findAll(ClassOrInterfaceDeclaration.class)
                            .stream()
                            .filter(ClassOrInterfaceDeclaration::isPublic)
                            .collect(Collectors.toList());
                    
                    for (ClassOrInterfaceDeclaration cls : classes) {
                        String className = cls.getNameAsString();
                        
                        List<MethodDeclaration> methods = cls.getMethods()
                                .stream()
                                .filter(MethodDeclaration::isPublic)
                                .collect(Collectors.toList());
                        
                        for (MethodDeclaration method : methods) {
                            String methodName = method.getNameAsString();
                            String fullKey = packageName + "." + className + "." + methodName;
                            
                            // 避免键冲突
                            int counter = 1;
                            String uniqueKey = fullKey;
                            while (apiMap.containsKey(uniqueKey)) {
                                uniqueKey = fullKey + "_" + counter++;
                            }
                            
                            apiMap.put(uniqueKey, extractMethodInfo(packageName, className, method, filePath));
                        }
                    }
                }
            } catch (Exception e) {
                System.err.println("Error processing file: " + filePath);
                e.printStackTrace();
            }
        }
        return apiMap;
    }

    private static MethodInfo extractMethodInfo(String packageName, String className, 
                                               MethodDeclaration method, Path filePath) {
        MethodInfo info = new MethodInfo();
        info.packageName = packageName;
        info.className = className;
        info.methodName = method.getNameAsString();
        info.signature = buildMethodSignature(method);
        
        MethodDeclaration cleanMethod = method.clone();
        
        removeAllComments(cleanMethod);
        
        info.sourceCode = cleanMethod.toString();
        info.docString = extractJavadoc(method);
        info.exampleCode = extractExamplesFromJavadoc(info.docString);
        info.sourcePath = filePath.toString();
        return info;
    }
    
    private static void removeAllComments(Node node) {
        node.getComment().ifPresent(Comment::remove);
        node.getAllContainedComments().forEach(Comment::remove);
        node.getChildNodes().forEach(SpringBootAPIExtractor::removeAllComments);
    }

    private static String buildMethodSignature(MethodDeclaration method) {
        String params = method.getParameters().stream()
                .map(p -> p.getType().asString() + " " + p.getNameAsString())
                .collect(Collectors.joining(", "));
        
        String returnType = method.getType().asString();
        return returnType + " " + method.getNameAsString() + "(" + params + ")";
    }

    private static String extractJavadoc(MethodDeclaration method) {
        return method.getJavadocComment()
                .map(comment -> comment.parse().toText())
                .orElse("");
    }

    private static String extractExamplesFromJavadoc(String javadoc) {
        if (javadoc.isEmpty()) return "";
        
        String[] lines = javadoc.split("\\R");
        StringBuilder example = new StringBuilder();
        boolean inExample = false;
        
        for (String line : lines) {
            if (line.trim().startsWith("@example")) {
                inExample = true;
                example.append(line.replace("@example", "").trim()).append("\n");
            } else if (inExample && line.trim().startsWith("@")) {
                inExample = false;
            } else if (inExample) {
                example.append(line).append("\n");
            }
        }
        
        return example.toString().trim();
    }

    private static List<Path> findJavaFiles(String directory) throws IOException {
        Path path = Paths.get(directory);
        
        if (!Files.exists(path)) {
            System.err.println("❌ Error: Path does not exist: " + path.toAbsolutePath());
            throw new IllegalArgumentException("Invalid source path: " + directory);
        }
        
        if (!Files.isDirectory(path)) {
            System.err.println("❌ Error: Path is not a directory: " + path.toAbsolutePath());
            throw new IllegalArgumentException("Path is not a directory: " + directory);
        }
        
        System.out.println("Scanning directory: " + path.toAbsolutePath());
        
        try (Stream<Path> paths = Files.walk(path)) {
            List<Path> javaFiles = paths
                    .filter(Files::isRegularFile)
                    .filter(p -> p.toString().endsWith(".java"))
                    .collect(Collectors.toList());
            
            System.out.println("Found " + javaFiles.size() + " Java files");
            return javaFiles;
        }
    }

    private static void writeToJson(Map<String, MethodInfo> apiMap, String outputFile) throws IOException {
        JsonObject jsonObject = new JsonObject();
        Gson gson = new GsonBuilder().setPrettyPrinting().create();

        for (Map.Entry<String, MethodInfo> entry : apiMap.entrySet()) {
            MethodInfo info = entry.getValue();
            
            JsonObject methodObj = new JsonObject();
            methodObj.addProperty("package", info.packageName);
            methodObj.addProperty("className", info.className);
            methodObj.addProperty("name", info.methodName);
            methodObj.addProperty("signature", info.signature);
            methodObj.addProperty("docString", info.docString);
            methodObj.addProperty("sourceCode", info.sourceCode);
            methodObj.addProperty("exampleCode", info.exampleCode);
            methodObj.addProperty("file", info.sourcePath);
            
            jsonObject.add(entry.getKey(), methodObj);
        }

        try (FileWriter writer = new FileWriter(outputFile)) {
            gson.toJson(jsonObject, writer);
        }
    }

    static class MethodInfo {
        String packageName;
        String className;
        String methodName;
        String signature;
        String docString;
        String sourceCode;
        String exampleCode;
        String sourcePath;
    }
}