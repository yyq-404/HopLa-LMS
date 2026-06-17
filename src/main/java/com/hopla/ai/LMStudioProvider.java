package com.hopla.ai;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.hopla.Completer;
import com.hopla.HopLa;
import okhttp3.Call;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

import static com.hopla.Constants.DEBUG_AI;

public class LMStudioProvider extends AIProvider {
    private final Map<AIChats.Chat, String> chatResponseIds = new ConcurrentHashMap<>();

    public LMStudioProvider(LLMConfig config, LLMConfig.Provider providerConfig) {
        super(AIProviderType.LMSTUDIO, AIProviderType.LMSTUDIO.toString(), config, providerConfig);
    }

    @Override
    public List<String> autoCompletion(Completer.CaretContext caretContext) throws IOException {
        List<String> completionParts = new ArrayList<>();
        if (providerConfig.completion_model == null || providerConfig.completion_model.isEmpty()) {
            throw new IOException("LM Studio completion model undefined");
        }

        JsonObject jsonPayload = new JsonObject();
        jsonPayload.addProperty("model", providerConfig.completion_model);
        jsonPayload.addProperty("prompt", promptReplace(caretContext, providerConfig.completion_prompt));
        jsonPayload.addProperty("stream", false);

        if (!providerConfig.completion_params.isEmpty()) {
            for (Map.Entry<String, Object> entry : providerConfig.completion_params.entrySet()) {
                if (entry.getValue() instanceof Number) {
                    jsonPayload.addProperty(entry.getKey(), (Number) entry.getValue());
                } else if (entry.getValue() instanceof Boolean) {
                    jsonPayload.addProperty(entry.getKey(), (Boolean) entry.getValue());
                } else {
                    jsonPayload.addProperty(entry.getKey(), String.valueOf(entry.getValue()));
                }
            }
        }

        if (!providerConfig.completion_stops.isEmpty()) {
            JsonArray stopArray = new JsonArray();
            providerConfig.completion_stops.forEach(stopArray::add);
            jsonPayload.add("stop", stopArray);
        }

        String jsonString = gson.toJson(jsonPayload);

        if (DEBUG_AI) {
            HopLa.montoyaApi.logging().logToOutput("LM Studio completion request: " + jsonString);
        }

        RequestBody body = RequestBody.create(jsonString, JSON);
        Request.Builder builder = new Request.Builder().url(providerConfig.completion_endpoint);

        for (Map.Entry<String, Object> entry : providerConfig.headers.entrySet()) {
            builder.addHeader(entry.getKey(), String.valueOf(entry.getValue()));
        }

        Request request = builder.post(body).build();

        if (currentCompletionCall != null) {
            currentCompletionCall.cancel();
        }
        currentCompletionCall = client.newCall(request);

        try (Response response = currentCompletionCall.execute()) {
            if (!response.isSuccessful()) {
                throw new IOException(response.code() + "\n" + Objects.requireNonNull(response.body()).string());
            }

            String responseBody = Objects.requireNonNull(response.body()).string();
            JsonObject responseJson = gson.fromJson(responseBody, JsonObject.class);

            // Handle OpenAI-compatible format
            if (responseJson.has("choices")) {
                JsonArray choices = responseJson.getAsJsonArray("choices");
                if (choices != null && !choices.isEmpty()) {
                    String text = choices.get(0).getAsJsonObject().get("text").getAsString();
                    if (text.contains(" ")) {
                        completionParts.add(text.split(" ")[0]);
                    }
                    completionParts.add(text);
                }
            }
            // Handle native LM Studio output format
            else if (responseJson.has("output")) {
                JsonArray outputArray = responseJson.getAsJsonArray("output");
                if (outputArray != null && !outputArray.isEmpty()) {
                    for (int i = 0; i < outputArray.size(); i++) {
                        JsonObject outputItem = outputArray.get(i).getAsJsonObject();
                        if (outputItem.has("type") && "message".equals(outputItem.get("type").getAsString())) {
                            String text = outputItem.get("content").getAsString();
                            if (text.contains(" ")) {
                                completionParts.add(text.split(" ")[0]);
                            }
                            completionParts.add(text);
                        }
                    }
                }
            }
        } catch (Exception e) {
            throw new IOException("LM Studio completion error: " + e.getMessage());
        }

        if (DEBUG_AI) {
            HopLa.montoyaApi.logging().logToOutput("LM Studio completions: " + completionParts);
        }
        return completionParts;
    }

    @Override
    public void instruct(String prompt, StreamingCallback callback) throws IOException {
        if (providerConfig.quick_action_model == null || providerConfig.quick_action_model.isEmpty()) {
            throw new IOException("LM Studio quick action model undefined");
        }

        JsonObject jsonPayload = new JsonObject();
        jsonPayload.addProperty("model", providerConfig.quick_action_model);
        jsonPayload.addProperty("stream", true);

        boolean isOpenAI = providerConfig.quick_action_endpoint != null
                && providerConfig.quick_action_endpoint.contains("/v1/chat/completions");

        if (isOpenAI) {
            // OpenAI format: expects messages array
            JsonArray messages = new JsonArray();
            if (!providerConfig.quick_action_system_prompt.isEmpty()) {
                JsonObject systemMsg = new JsonObject();
                systemMsg.addProperty("role", AIChats.MessageRole.SYSTEM.toString().toLowerCase());
                systemMsg.addProperty("content", providerConfig.quick_action_system_prompt);
                messages.add(systemMsg);
            }
            JsonObject userMsg = new JsonObject();
            userMsg.addProperty("role", AIChats.MessageRole.USER.toString().toLowerCase());
            userMsg.addProperty("content", prompt);
            messages.add(userMsg);
            jsonPayload.add("messages", messages);
        } else {
            // LM Studio native format: expects input
            jsonPayload.addProperty("input", prompt);
            if (!providerConfig.quick_action_system_prompt.isEmpty()) {
                jsonPayload.addProperty("system_prompt", providerConfig.quick_action_system_prompt);
            }
        }

        if (!providerConfig.quick_action_params.isEmpty()) {
            for (Map.Entry<String, Object> entry : providerConfig.quick_action_params.entrySet()) {
                if (entry.getValue() instanceof Number) {
                    jsonPayload.addProperty(entry.getKey(), (Number) entry.getValue());
                } else if (entry.getValue() instanceof Boolean) {
                    jsonPayload.addProperty(entry.getKey(), (Boolean) entry.getValue());
                } else {
                    jsonPayload.addProperty(entry.getKey(), String.valueOf(entry.getValue()));
                }
            }
        }

        if (!providerConfig.quick_action_stops.isEmpty()) {
            JsonArray stopArray = new JsonArray();
            providerConfig.quick_action_stops.forEach(stopArray::add);
            jsonPayload.add("stop", stopArray);
        }

        String jsonString = gson.toJson(jsonPayload);

        if (DEBUG_AI) {
            HopLa.montoyaApi.logging().logToOutput("LM Studio instruct request: " + jsonString);
        }

        RequestBody body = RequestBody.create(jsonString, JSON);
        Request.Builder builder = new Request.Builder().url(providerConfig.quick_action_endpoint);

        for (Map.Entry<String, Object> entry : providerConfig.headers.entrySet()) {
            builder.addHeader(entry.getKey(), String.valueOf(entry.getValue()));
        }

        Request request = builder.post(body).build();

        if (currentQuickActionCall != null) {
            currentQuickActionCall.cancel();
        }
        currentQuickActionCall = client.newCall(request);
        sendStreamingRequest(currentQuickActionCall, callback, null);
    }

    @Override
    public void chat(AIChats.Chat chat, StreamingCallback callback) throws IOException {
        if (providerConfig.chat_model == null || providerConfig.chat_model.isEmpty()) {
            throw new IOException("LM Studio chat model undefined");
        }

        JsonObject jsonPayload = new JsonObject();
        jsonPayload.addProperty("model", providerConfig.chat_model);
        jsonPayload.addProperty("stream", true);

        boolean isOpenAI = providerConfig.chat_endpoint != null
                && providerConfig.chat_endpoint.contains("/v1/chat/completions");

        if (isOpenAI) {
            // OpenAI format: expects full messages history
            JsonArray messages = new JsonArray();
            if (!providerConfig.chat_system_prompt.isEmpty()) {
                JsonObject systemMsg = new JsonObject();
                systemMsg.addProperty("role", AIChats.MessageRole.SYSTEM.toString().toLowerCase());
                systemMsg.addProperty("content", providerConfig.chat_system_prompt);
                messages.add(systemMsg);
            }

            // subList to exclude the last empty assistant response message
            for (AIChats.Message message : chat.getMessages().subList(0, chat.getMessages().size() - 1)) {
                JsonObject userMessage = new JsonObject();
                userMessage.addProperty("role", message.getRole().toString().toLowerCase());
                userMessage.addProperty("content", message.getContent());
                messages.add(userMessage);
            }
            jsonPayload.add("messages", messages);
        } else {
            // LM Studio native format: expects single input and previous_response_id
            int messagesSize = chat.getMessages().size();
            if (messagesSize < 2) {
                HopLa.montoyaApi.logging().logToOutput("No user message in chat");
                return;
            }
            // Second to last is the user prompt (last one is the empty assistant message)
            String prompt = chat.getMessages().get(messagesSize - 2).getContent();
            jsonPayload.addProperty("input", prompt);

            if (!providerConfig.chat_system_prompt.isEmpty()) {
                jsonPayload.addProperty("system_prompt", providerConfig.chat_system_prompt);
            }

            // Include previous_response_id if available to maintain conversation context
            String previousResponseId = chatResponseIds.get(chat);
            if (previousResponseId != null && messagesSize > 2) {
                jsonPayload.addProperty("previous_response_id", previousResponseId);
            }
        }

        if (!providerConfig.chat_params.isEmpty()) {
            for (Map.Entry<String, Object> entry : providerConfig.chat_params.entrySet()) {
                if (entry.getValue() instanceof Number) {
                    jsonPayload.addProperty(entry.getKey(), (Number) entry.getValue());
                } else if (entry.getValue() instanceof Boolean) {
                    jsonPayload.addProperty(entry.getKey(), (Boolean) entry.getValue());
                } else {
                    jsonPayload.addProperty(entry.getKey(), String.valueOf(entry.getValue()));
                }
            }
        }

        if (!providerConfig.chat_stops.isEmpty()) {
            JsonArray stopArray = new JsonArray();
            providerConfig.chat_stops.forEach(stopArray::add);
            jsonPayload.add("stop", stopArray);
        }

        String jsonString = gson.toJson(jsonPayload);

        if (DEBUG_AI) {
            HopLa.montoyaApi.logging().logToOutput("LM Studio chat request: " + jsonString);
        }

        RequestBody body = RequestBody.create(jsonString, JSON);
        Request.Builder builder = new Request.Builder().url(providerConfig.chat_endpoint);

        for (Map.Entry<String, Object> entry : providerConfig.headers.entrySet()) {
            builder.addHeader(entry.getKey(), String.valueOf(entry.getValue()));
        }

        Request request = builder.post(body).build();

        if (currentChatcall != null) {
            currentChatcall.cancel();
        }
        currentChatcall = client.newCall(request);
        sendStreamingRequest(currentChatcall, callback, chat);
    }

    private void sendStreamingRequest(Call call, StreamingCallback callback, AIChats.Chat associatedChat) {
        new Thread(() -> {
            try (Response response = call.execute()) {
                if (!response.isSuccessful()) {
                    callback.onError("LM Studio API error: " + response.code() + "\n"
                            + Objects.requireNonNull(response.body()).string());
                    return;
                }

                BufferedReader reader = new BufferedReader(
                        new InputStreamReader(response.body().byteStream(), StandardCharsets.UTF_8));
                String line;
                while ((line = reader.readLine()) != null) {
                    if (Thread.currentThread().isInterrupted())
                        break;

                    if (!line.startsWith("data: "))
                        continue;

                    String jsonLine = line.substring("data: ".length());
                    if (jsonLine.isBlank() || "[DONE]".equals(jsonLine.trim()))
                        continue;

                    JsonObject responseJson = gson.fromJson(jsonLine, JsonObject.class);

                    // 1. Handle native LM Studio format (SSE with event type/data)
                    if (responseJson.has("type")) {
                        String eventType = responseJson.get("type").getAsString();
                        if ("message.delta".equals(eventType) || "reasoning.delta".equals(eventType)) {
                            if (responseJson.has("content")) {
                                String content = responseJson.get("content").getAsString();
                                if (DEBUG_AI) {
                                    HopLa.montoyaApi.logging().logToOutput("LM Studio streaming response: " + content);
                                }
                                callback.onData(content);
                            }
                        } else if ("chat.end".equals(eventType)) {
                            if (responseJson.has("result")) {
                                JsonObject result = responseJson.getAsJsonObject("result");
                                if (result != null && result.has("response_id")) {
                                    String responseId = result.get("response_id").getAsString();
                                    if (associatedChat != null) {
                                        chatResponseIds.put(associatedChat, responseId);
                                    }
                                }
                            }
                        } else if ("error".equals(eventType)) {
                            JsonObject errorObj = responseJson.getAsJsonObject("error");
                            String errorMsg = errorObj != null && errorObj.has("message")
                                    ? errorObj.get("message").getAsString()
                                    : "Unknown LM Studio error";
                            callback.onError("LM Studio error: " + errorMsg);
                            return;
                        }
                    }
                    // 2. Handle OpenAI-compatible format
                    else if (responseJson.has("choices")) {
                        JsonArray choices = responseJson.getAsJsonArray("choices");
                        if (choices != null && !choices.isEmpty()) {
                            JsonObject choice = choices.get(0).getAsJsonObject();
                            if (choice.has("delta")) {
                                JsonObject delta = choice.getAsJsonObject("delta");
                                if (delta.has("content")) {
                                    String content = delta.get("content").getAsString();
                                    if (DEBUG_AI) {
                                        HopLa.montoyaApi.logging().logToOutput(
                                                "LM Studio OpenAI-compatible streaming response: " + content);
                                    }
                                    callback.onData(content);
                                }
                            }
                        }
                    }
                }
                callback.onDone();
            } catch (IOException ex) {
                callback.onError("Cancelled or error: " + ex.getMessage());
            } catch (Exception ex) {
                callback.onError("LM Studio streaming error: " + ex.getMessage());
            }
        }).start();
    }
}
