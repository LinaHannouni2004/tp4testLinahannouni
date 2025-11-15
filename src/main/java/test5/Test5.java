package test5;

import assistant.Assistant;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class Test5 {

    public static void main(String[] args) throws Exception {

        // ---- 1) MODEL ----
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-2.5-flash")
                .logRequestsAndResponses(true)
                .build();

        // ---- 2) EMBEDDING MODEL ----
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // ---- 3) LOAD PDF + INDEX ----
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        ingestPdf("rag.pdf", store, embeddingModel);

        ContentRetriever pdfRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .build();

        // ---- 4) TAVILY WEB SEARCH ----
        WebSearchEngine tavily = TavilyWebSearchEngine.builder()
                .apiKey(System.getenv("TAVILY_API_KEY"))
                .build();

        ContentRetriever webRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(tavily)
                .build(); 

        // ---- 5) ROUTAGE PDF + WEB ----
        DefaultQueryRouter router = new DefaultQueryRouter(
                List.of(pdfRetriever, webRetriever)
        );

        // ---- 6) RAG AUGMENTOR ----
        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        // ---- 7) ASSISTANT ----
        Assistant assistant = dev.langchain4j.service.AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(augmentor)
                .build();

        // ---- 8) CHAT LOOP ----
        Scanner scanner = new Scanner(System.in);
        System.out.println("Test 5 - RAG + WebSearch Tavily activé.");

        while (true) {
            System.out.print("\nVous > ");
            String question = scanner.nextLine();
            if (question.equalsIgnoreCase("exit")) break;

            System.out.println("\nAssistant > " + assistant.chat(question));
        }
    }


    private static void ingestPdf(String fileName,
                                  EmbeddingStore<TextSegment> store,
                                  EmbeddingModel embeddingModel) throws Exception {

        Path path = getResourcePath(fileName);
        Document doc = FileSystemDocumentLoader.loadDocument(path, new ApacheTikaDocumentParser());

        var splitter = DocumentSplitters.recursive(300, 30);
        var segments = splitter.split(doc);

        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        store.addAll(embeddings, segments);
    }

    private static Path getResourcePath(String name) throws URISyntaxException {
        return Paths.get(Objects.requireNonNull(
                Test5.class.getClassLoader().getResource(name)
        ).toURI());
    }
}
