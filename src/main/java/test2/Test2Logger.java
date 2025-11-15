package test2;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.embedding.Embedding;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;

import dev.langchain4j.service.AiServices;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStore;

import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;

import assistant.Assistant;
import test1.rag.RagNaif;

import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;

import java.util.List;
import java.util.Objects;
import java.util.Scanner;

import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test2Logger {

    public static void main(String[] args) throws Exception {

        // ---- Logging LangChain4j ----
        configureLogger();

        // ---- Modèle Gemini ----
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-2.5-flash")
                .temperature(0.2)
                .logRequestsAndResponses(true)
                .build();

        // ---- Charger PDF ----
        Path pdfPath = getResourcePath("rag.pdf");
        ApacheTikaDocumentParser parser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(pdfPath, parser);

        // ---- Découpage du document ----
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        System.out.println("\n===== SEGMENTS DU DOCUMENT (PHASE 1) =====");
        int i = 1;
        for (TextSegment s : segments) {
            System.out.println("---- SEGMENT " + i++ + " ----");
            System.out.println(s.text());
            System.out.println("----------------------------------\n");
        }

        // ---- Embeddings ----
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        // ---- Store ----
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

        // ---- Retriever pour l'assistant ----
        var retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingModel(embeddingModel)
                .embeddingStore(store)
                .maxResults(2)
                .minScore(0.5)
                .build();

        var memory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(memory)
                .contentRetriever(retriever)
                .build();

        // ---- Boucle utilisateur ----
        try (Scanner scanner = new Scanner(System.in)) {
            System.out.println("Test 2 - Logging activé. Tape une question :");

            while (true) {
                System.out.print("\nVous > ");
                String question = scanner.nextLine();
                if (question == null || question.equalsIgnoreCase("exit")) break;

                System.out.println("\n===== SEGMENTS RETROUVÉS (MANUEL, PHASE 2) =====");

                // 1. Embedding de la question
                Embedding embeddingQuestion = embeddingModel.embed(question).content();

                // 2. Construire la requête de recherche
                EmbeddingSearchRequest request = EmbeddingSearchRequest.builder()
                        .queryEmbedding(embeddingQuestion)
                        .maxResults(3)
                        .minScore(0.5)
                        .build();

                // 3. Récupérer les résultats
                EmbeddingSearchResult<TextSegment> result = store.search(request);

                // 4. Affichage des segments avec score
                for (EmbeddingMatch<TextSegment> match : result.matches()) {
                    System.out.println("Score : " + match.score());
                    System.out.println("Segment : " + match.embedded().text());
                    System.out.println("-------------------------------------------");
                }

                // ---- Réponse finale du LLM ----
                System.out.println("\nAssistant > " + assistant.chat(question));
            }
        }
    }

    // ---- Logger ----
    private static void configureLogger() {
        Logger logger = Logger.getLogger("dev.langchain4j");
        logger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        logger.addHandler(handler);
    }

    // ---- Charger un fichier depuis /resources ----
    private static Path getResourcePath(String name) {
        try {
            var url = Objects.requireNonNull(
                    RagNaif.class.getClassLoader().getResource(name),
                    "Fichier introuvable : " + name
            );
            return Paths.get(url.toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }
}
