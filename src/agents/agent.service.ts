import { Injectable } from '@nestjs/common';
import { StateGraph, START, END } from '@langchain/langgraph';
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { AIMessage, HumanMessage, SystemMessage, BaseMessage } from "@langchain/core/messages";
import * as fs from 'fs';
import * as path from 'path';
import * as AWS from 'aws-sdk';
import * as conversationFs from 'fs';
import { z } from 'zod';
import { DeployTool } from './deploy.tool';

interface CloudState {
  conversationHistory: any[];
  query: string;
  action: any;
  csp: string | null;
  response: any;
  finalResponse?: any;
  event?: string;
}

interface ServiceConfig {
  name: string;
  description: string;
  cloud: string;
  requiredFields: Array<{
    type: string;
    fieldId: string;
    fieldName: string;
    fieldValue: string;
    fieldTypeValue: string;
  }>;
}

interface ServicesData {
  list: ServiceConfig[];
}

@Injectable()
export class AgentService {
  private workflow: any;
  private readonly llm: ChatGoogleGenerativeAI;
  private conversations: Map<string, any> = new Map();
  private servicesData: ServicesData | null = null;
  private conversationFilePath = path.join(process.cwd(), 'conversations.json');
  private deployTool: DeployTool;

  constructor() {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error('GEMINI_API_KEY environment variable is not set');
    }
    
    // Initialize LLM
    this.llm = new ChatGoogleGenerativeAI({
      apiKey,
      model: "gemini-1.5-flash",
      temperature: 0.2,
      maxOutputTokens: 2048,
      json: true
    });
    
    this.loadServicesData();
    this.deployTool = new DeployTool();

    // Load all conversations from file into memory
    const allConversations = this.loadAllConversationsFromFile();
    Object.entries(allConversations).forEach(([userId, convo]) => {
      this.conversations.set(userId, convo);
    });

    // LangGraph workflow setup - remove analyzeInput
    const CloudStateSchema = z.object({
      conversationHistory: z.array(z.any()),
      query: z.string(),
      action: z.any(),
      csp: z.string().nullable(),
      response: z.any(),
      finalResponse: z.any().nullable(),
    });
    
    const workflow = new StateGraph(CloudStateSchema)
      .addNode('findAction', async (state: CloudState) => await this.findAction(state))
      .addNode('processAction', async (state: CloudState) => await this.processAction(state))
      .addNode('generateUnifiedResponse', async (state: CloudState) => await this.generateUnifiedResponse(state));
    
    workflow.addEdge(START, 'findAction');
    workflow.addEdge('findAction', 'processAction');
    workflow.addEdge('processAction', 'generateUnifiedResponse');
    workflow.addEdge('generateUnifiedResponse', END);
    
    this.workflow = workflow.compile();
  }

  private loadServicesData() {
    try {
      const servicesPath = path.join(process.cwd(), 'services.json');
      const servicesContent = fs.readFileSync(servicesPath, 'utf8');
      this.servicesData = JSON.parse(servicesContent);
    } catch (error) {
      console.error('Error loading services.json:', error);
      this.servicesData = null;
    }
  }

  private findMatchingService(analysis: any): ServiceConfig | null {
    if (!this.servicesData || !analysis) return null;

    const { details, csp } = analysis;
    let serviceName = typeof details?.service === 'string' ? details.service : null;
    const cloudProvider = typeof csp === 'string' ? csp.toLowerCase() : null;

    // Normalize: trim, lowercase, collapse multiple spaces
    const normalize = (str: string) => str.trim().toLowerCase().replace(/\s+/g, ' ');
    serviceName = serviceName ? normalize(serviceName) : null;
    if (this.servicesData) {
      this.servicesData.list.forEach(service => {
        console.log("Available service:", normalize(service.name), "cloud:", service.cloud.toLowerCase());
      });
    }

    if (!serviceName || !cloudProvider) return null;

    // Find matching service by normalized name and cloud provider
    return this.servicesData.list.find(service =>
      typeof service.name === 'string' &&
      normalize(service.name) === serviceName &&
      typeof service.cloud === 'string' &&
      service.cloud.toLowerCase() === cloudProvider
    ) || null;
  }

  private cleanJsonResponse(response: string): string {
    // Remove markdown code block formatting
    return response.replace(/```json\n?|\n?```/g, '').trim();
  }

  private async findAction(cloudState: CloudState): Promise<CloudState> {
    const { query, conversationHistory, csp } = cloudState;
    console.log("Generating findAction response");

    // Create a comprehensive unified prompt
    const promptText = `You're a cloud deployment assistant that helps users provision cloud resources and answer general questions about cloud services. 
    You need to determine the appropriate action for this conversation step.

    AVAILABLE ACTIONS:
    
    📋 Cloud Management Actions:
    - "GENERAL_RESPONSE" - User asks a general question about cloud services or concepts
    - "DEPLOY" - User wants to deploy or provision a specific cloud service
    - "CONFIGURE" - User wants to configure an existing cloud service
    - "LIST_RESOURCES" - User wants to list or view their cloud resources
    - "VIEW_CSP_OPTIONS" - User wants to see cloud service provider options
    - "SELECT_CSP" - User selects a specific Cloud Service Provider (AWS, Azure, GCP, etc.)
    - "CONVERSATION_SUMMARY" - User asks about conversation history or previous questions
    
    Rules:
    - Return action always in the response from the available actions list
    - If the user is asking about how many questions they've asked, use CONVERSATION_SUMMARY with questionCount
    - If the user is asking about what they've previously discussed or asked, use CONVERSATION_SUMMARY
    - If the user is asking a general question about cloud services, use GENERAL_RESPONSE
    - If the user mentions deploying, creating, or provisioning a service, use DEPLOY
    - If the user wants to see available services, use VIEW_CSP_OPTIONS
    - If the user is explicitly selecting a cloud provider, use SELECT_CSP
    - Extract as much information as possible from the query to fill the payload
    
    Return JSON only in this format:
    {
      "action": {
        "type": "ACTION_TYPE", // ACTION_TYPE is one of the actions defined above
        "payload": {
          "service": "service-name", // Name of the cloud service the user wants to deploy
          "csp": "aws|azure|gcp|oracle", // The cloud service provider mentioned
          "region": "region-name", // The region mentioned, if any
          "specifications": {}, // Any additional specifications mentioned
          "message": "user's message simplified", // A simplified version of the user's message
          "questionCount": 5 // Include only for CONVERSATION_SUMMARY when asking about question counts
        }
      }
    }`;

    // Generate conversation history messages from the history
    const messagePayload = [
      new SystemMessage(promptText),
      ...this.generateConversationHistory(conversationHistory),
      new HumanMessage(String(query))
    ];

    console.log("Sending payload to LLM for action determination");
    try {
      console.log(`Processing query: "${query}" for CSP: ${csp || 'unknown'}`);
      const response = await this.llm.invoke(messagePayload);
      const responseContent = String(response.content);
      
      try {
        const parsedResponse = JSON.parse(responseContent);
        console.log("parsedResponse from findAction:", JSON.stringify(parsedResponse, null, 2));
        
        // Handle the case where the LLM returns a complete response object instead of just the action
        if (parsedResponse.role === 'assistant' && parsedResponse.workflow && parsedResponse.response) {
          console.log("LLM returned a complete response object instead of just the action");
          // Extract the action type from the workflow if possible
          let actionType = "GENERAL_RESPONSE"; 
          if (parsedResponse.workflow === "conversationSummary") {
            actionType = "CONVERSATION_SUMMARY";
          } else if (parsedResponse.workflow === "deployService") {
            actionType = "DEPLOY";
          } else if (parsedResponse.workflow === "configureService") {
            actionType = "CONFIGURE";
          } else if (parsedResponse.workflow === "listResources") {
            actionType = "LIST_RESOURCES";
          } else if (parsedResponse.workflow === "viewCspOptions") {
            actionType = "VIEW_CSP_OPTIONS";
          } else if (parsedResponse.workflow === "selectCsp") {
            actionType = "SELECT_CSP";
          }
          
          // Extract questionCount if this is a conversation summary
          let questionCount;
          if (actionType === "CONVERSATION_SUMMARY") {
            // Try to extract question count from message
            const countMatch = parsedResponse.response.message?.match(/(\d+)\s+questions?/i);
            if (countMatch && countMatch[1]) {
              questionCount = parseInt(countMatch[1], 10);
            }
            
            // If we have previousQuestions array, use its length as a fallback
            if (!questionCount && Array.isArray(parsedResponse.response.previousQuestions)) {
              questionCount = parsedResponse.response.previousQuestions.length;
            }
          }
          
          // Create a synthetic action object
          const actionObject = {
            type: actionType,
            payload: {
              service: parsedResponse.response.service,
              csp: parsedResponse.response.csp || csp,
              region: parsedResponse.response.region,
              specifications: parsedResponse.response.specifications || {},
              message: parsedResponse.response.message,
              questionCount
            }
          };
          
          // Set CSP from response if available
          let updatedCsp = csp;
          if (parsedResponse.response.csp) {
            updatedCsp = parsedResponse.response.csp.toLowerCase();
          }
          
          return {
            ...cloudState,
            action: actionObject,
            csp: updatedCsp
          };
        }
        
        // Handle the expected action format
        if (parsedResponse.action) {
          // Set CSP from response if available
          let updatedCsp = csp;
          if (parsedResponse.action.payload?.csp) {
            updatedCsp = parsedResponse.action.payload.csp.toLowerCase();
          }
          
          return {
            ...cloudState,
            action: parsedResponse.action,
            csp: updatedCsp
          };
        } else {
          // Fallback to a general response with the original query
          console.log("Response did not contain expected 'action' property:", parsedResponse);
          return {
            ...cloudState,
            action: {
              type: "GENERAL_RESPONSE",
              payload: {
                message: query
              }
            }
          };
        }
      } catch (jsonError) {
        console.error("JSON parse error in findAction:", jsonError);
        console.log("Raw content:", responseContent);
        
        // Fallback to a general response with the original query
        return {
          ...cloudState,
          action: {
            type: "GENERAL_RESPONSE",
            payload: {
              message: query
            }
          }
        };
      }
    } catch (e) {
      console.error("Error in findAction LLM invocation:", e);
      return {
        ...cloudState,
        action: {
          type: "GENERAL_RESPONSE",
          payload: {
            message: query
          }
        }
      };
    }
  }

  // Helper method to generate conversation history
  private generateConversationHistory(history: any[]): BaseMessage[] {
    if (!history || history.length === 0) return [];
    
    const messages: BaseMessage[] = [];
    
    for (const msg of history) {
      if (msg.role === 'human') {
        messages.push(new HumanMessage(msg.content));
      } else if (msg.role === 'assistant') {
        // Handle complex content objects
        const content = typeof msg.content === 'object' 
          ? JSON.stringify(msg.content)
          : String(msg.content);
        messages.push(new AIMessage(content));
      }
    }
    
    return messages;
  }

  private async generateExampleValues(service: any, csp: string): Promise<any> {
    const promptText = `Generate example values and explanations for the following cloud service configuration fields.
    
    Service: ${service.name}
    Cloud Provider: ${csp.toUpperCase()}
    Fields: ${JSON.stringify(service.requiredFields)}
    
    Return JSON only:
    {
      "fields": [
        {
          "fieldId": "string",
          "exampleValue": "string",
          "explanation": "string explaining the example value and best practices"
        }
      ]
    }

    Guidelines:
    - Provide realistic, production-ready example values
    - Follow cloud provider best practices
    - Consider security and naming conventions
    - Values should be valid for the specified field types
    - Include brief explanation of why these values are good examples
    - Keep explanations concise but informative`;

    try {
      console.log(`Generating example values for ${service.name} on ${csp}`);
      const messagePayload = [
        new SystemMessage(promptText),
        new HumanMessage(promptText)
      ];
      const result = await this.llm.invoke(messagePayload);
      const responseText = String(result.content);
      console.log("Example values generation complete");

      try {
        const cleanedResponse = this.cleanJsonResponse(responseText);
        const parsed = JSON.parse(cleanedResponse);
        return parsed;
      } catch (e) {
        console.error("Error parsing example values:", e);
        // Provide default empty fields if parsing fails
        return {
          fields: service.requiredFields.map(field => ({
            fieldId: field.fieldId,
            exampleValue: "",
            explanation: "Please provide a value appropriate for this field."
          }))
        };
      }
    } catch (e) {
      console.error("Error generating example values:", e);
      // Provide default empty fields on error
      return {
        fields: service.requiredFields.map(field => ({
          fieldId: field.fieldId,
          exampleValue: "",
          explanation: "Please provide a value appropriate for this field."
        }))
      };
    }
  }

  private async processAction(cloudState: CloudState): Promise<CloudState> {
    const { action, csp, conversationHistory, response } = cloudState;
    console.log("Processing action:", action?.type);
    
    if (!action) {
      return cloudState;
    }

    let customResponse = "";
    let updatedState = { ...cloudState };
    
    switch (action.type) {
      case 'GENERAL_RESPONSE':
        // For general questions, we'll let the generateUnifiedResponse handle it
        break;
        
      case 'CONVERSATION_SUMMARY':
        try {
          // Check if the action payload already contains a well-formed message from the LLM
          if (action.payload.message && !action.payload.message.startsWith("You have asked") && !action.payload.message.includes("summary")) {
            // Use the message from the LLM directly
            customResponse = action.payload.message;
          } else {
            // Fallback to checking if it's a count query or summary query
            const isCountQuery = /how many|number of/i.test(action.payload.message || cloudState.query);
            const questionCount = action.payload.questionCount;
            
            if (isCountQuery) {
              // For count queries, provide a direct answer with just the number
              const count = questionCount || (
                Array.isArray(conversationHistory)
                  ? conversationHistory.filter(msg => msg.role === 'human').length - 1
                  : 0
              );
              customResponse = `You have asked ${count} questions so far.`;
            } else {
              // For summary queries, provide an introduction to the list
              customResponse = "Here's a summary of our conversation so far:";
            }
          }
          
          updatedState.event = "conversation_summarized";
          
          // Check if we already have a response with previousQuestions from the LLM
          if (response && response.previousQuestions && Array.isArray(response.previousQuestions)) {
            console.log("Using existing previousQuestions from response");
            
            // If the LLM response has a message, prefer it over our custom message
            const finalMessage = response.message || customResponse;
            
            updatedState.response = {
              ...response,
              status: "conversation_summarized",
              message: finalMessage
            };
            break;
          }
          
          // Extract only unique human messages and filter out the current question
          const humanMessages = [...new Set(conversationHistory
            .filter(msg => msg.role === 'human')
            .map(msg => typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content))
          )];
          
          // Remove the current question (it will be the last one)
          const previousQuestions = humanMessages.slice(0, -1);
          const actualCount = previousQuestions.length;
          
          // Don't override the LLM's message if it provided one
          if (!customResponse) {
            customResponse = action.payload.message || 
                            (actualCount > 0 
                              ? `You have asked ${actualCount} questions so far.` 
                              : "This is your first question.");
          }
          
          updatedState.response = {
            status: "conversation_summarized",
            message: customResponse,
            previousQuestions,
            questionCount: action.payload.questionCount || actualCount
          };
        } catch (error) {
          console.error("Error processing conversation summary:", error);
          customResponse = "I apologize, but I encountered an error retrieving your conversation history.";
          updatedState.response = {
            status: "error",
            message: customResponse
          };
        }
        break;
        
      case 'DEPLOY':
        const { service } = action.payload;
        if (service) {
          // Find the matching service in the services data
          const matchingService = this.findMatchingServiceByName(service, csp);
          if (matchingService) {
            // Generate example values for the service
            const exampleValues = await this.generateExampleValues(matchingService, csp || 'aws');
            
            // Enhance required fields with examples
            const enhancedFields = matchingService.requiredFields.map(field => {
              const exampleField = exampleValues.fields.find((ef: any) => ef.fieldId === field.fieldId);
              return {
                ...field,
                exampleValue: exampleField?.exampleValue || '',
                explanation: exampleField?.explanation || ''
              };
            });

            // Set the response
            updatedState.response = {
              status: "service_found",
              message: `Found service configuration for ${matchingService.name} on ${matchingService.cloud.toUpperCase()}`,
              service: {
                name: matchingService.name,
                description: matchingService.description,
                cloud: matchingService.cloud,
                requiredFields: enhancedFields
              },
              details: action.payload
            };
            
            updatedState.event = "service_configuration";
          } else {
            // Service not found
            const availableServices = this.servicesData?.list
              .filter(s => s.cloud === (csp || 'aws').toLowerCase())
              .map(s => s.name) || [];
              
            customResponse = `Sorry, the ${service} service is currently not available to provision in ${(csp || 'aws').toUpperCase()}. Please select from the available services.`;
            updatedState.event = "service_not_found";
            updatedState.response = {
              status: "service_not_found",
              availableServices,
              message: customResponse
            };
          }
        }
        break;
        
      case 'LIST_RESOURCES':
        const { resourceType } = action.payload;
        if (resourceType && resourceType.toLowerCase().includes('security group') && csp === 'aws') {
          const sgList = await this.listAWSSecurityGroups();
          customResponse = sgList;
          updatedState.event = "resources_listed";
          updatedState.response = {
            status: "resources_listed",
            message: sgList,
            resourceType
          };
        }
        break;
        
      case 'SELECT_CSP':
        const selectedCsp = action.payload.csp;
        if (selectedCsp) {
          updatedState.csp = selectedCsp.toLowerCase();
          customResponse = `You have selected ${selectedCsp.toUpperCase()} as your Cloud Service Provider.`;
          updatedState.event = "csp_selected";
          updatedState.response = {
            status: "csp_selected",
            message: customResponse,
            csp: selectedCsp.toLowerCase()
          };
        }
        break;
        
      case 'VIEW_CSP_OPTIONS':
        const cspOptions = ['AWS', 'Azure', 'GCP', 'Oracle Cloud'];
        const availableServicesPerCsp = {};
        
        if (this.servicesData) {
          cspOptions.forEach(cspOption => {
            const cspLower = cspOption.toLowerCase().replace(' cloud', '');
            const services = this.servicesData?.list
              .filter(s => s.cloud === cspLower)
              .map(s => s.name);
              
            availableServicesPerCsp[cspOption] = services;
          });
        }
        
        customResponse = `Here are the available cloud service providers:`;
        updatedState.event = "csp_options_shown";
        updatedState.response = {
          status: "csp_options_shown",
          message: customResponse,
          cspOptions,
          availableServicesPerCsp
        };
        break;
        
      default:
        break;
    }
    
    // Only set customResponse if it is not empty
    if (customResponse) {
      updatedState.response.customMessage = customResponse;
    }
    
    return updatedState;
  }
  
  // Helper method to find a service by name
  private findMatchingServiceByName(serviceName: string, csp: string | null): ServiceConfig | null {
    if (!this.servicesData) return null;
    
    // Normalize names for comparison
    const normalize = (str: string) => str.trim().toLowerCase().replace(/\s+/g, ' ');
    const normalizedServiceName = normalize(serviceName);
    const normalizedCsp = csp ? normalize(csp) : 'aws';
    
    // Find matching service
    return this.servicesData.list.find(service => 
      normalize(service.name).includes(normalizedServiceName) && 
      service.cloud.toLowerCase() === normalizedCsp
    ) || null;
  }

  private async generateUnifiedResponse(cloudState: CloudState): Promise<CloudState> {
    const { action, response, query, conversationHistory, csp } = cloudState;
    console.log("Generating unified response");
    
    // Check if user is asking about conversation history
    const isConversationHistoryQuery = /what.*asked|conversation.*history|previous.*questions|questions.*so far|history|recap|summarize.*conversation|how many.*questions|number of questions/i.test(query);
    
    // If the query is about conversation history, make sure we set the action type correctly
    if (isConversationHistoryQuery && (!action || action.type !== "CONVERSATION_SUMMARY")) {
      cloudState.action = {
        type: "CONVERSATION_SUMMARY",
        payload: {
          message: query
        }
      };
      // Process the action again with the corrected type
      return this.processAction(cloudState);
    }
    
    // If we already have a processed response for conversation summary, just return it
    if (action && action.type === "CONVERSATION_SUMMARY" && response && 
        (response.previousQuestions || response.questionCount)) {
      console.log("Using already processed conversation summary response");
      return {
        ...cloudState,
        finalResponse: {
          role: "assistant",
          workflow: "conversationSummary",
          response: {
            ...response,
            menu: this.getMenuForCSP(csp || 'aws', true)
          }
        }
      };
    }
    
    // Check if we have a custom message from processAction
    if (response && response.customMessage) {
      console.log("Using custom message from processAction");
      return {
        ...cloudState,
        finalResponse: {
          role: "assistant",
          workflow: response.status || "customResponse",
          response: {
            message: response.customMessage,
            ...response,
            menu: this.getMenuForCSP(csp || 'aws', true)
          }
        }
      };
    }
    
    // Check if we have a service configuration response
    if (response && response.status === 'service_found') {
      console.log("Using service_found response");
      return {
        ...cloudState,
        finalResponse: {
          role: "assistant",
          workflow: "serviceConfiguration",
          response: {
            message: `To provision ${response.service.name} on ${response.service.cloud.toUpperCase()}, please provide the following information:`,
            service: response.service,
            details: response.details,
            menu: this.getMenuForCSP(csp || 'aws', true)
          }
        }
      };
    }
    
    // Special prompt for conversation history queries
    let promptText;
    if (isConversationHistoryQuery) {
      promptText = `You are a cloud deployment assistant. The user is asking about their conversation history with you.
      
      Review the conversation history and provide a detailed response based on the type of query:
      
      1. If they're asking about how many questions they've asked:
         - Provide the exact number
         - Be conversational but direct (e.g., "You've asked 5 questions so far.")
         
      2. If they're asking to see their previous questions:
         - List all their previous questions in chronological order
         - Number the questions
         - Don't include the current question
         
      3. If they're asking for a general summary:
         - Summarize the main topics and questions they've asked about
         - Highlight key points from the conversation
      
      IMPORTANT: Return a complete response object in JSON:
      {
        "role": "assistant",
        "workflow": "conversationSummary",
        "response": {
          "message": "Your direct answer to their query",
          "previousQuestions": ["List", "of", "their", "previous", "questions"],
          "questionCount": 5,
          "csp": "${csp || 'aws'}"
        }
      }`;
    } else {
      // Standard prompt for regular queries
      promptText = `You're a cloud deployment assistant that helps users provision cloud resources and answer general questions about cloud services.
      You need to generate a helpful response.
      
      CURRENT CONVERSATION STATE:
      - Cloud Provider: ${csp ? csp.toUpperCase() : 'AWS'}
      - Action Type: ${action?.type || 'GENERAL_RESPONSE'}
      
      Rules:
      - Answer the user's question in a conversational manner and follow the conversation history
      - Never hallucinate. If you don't know the answer, say you don't know in a polite manner
      - Keep the response short and concise
      - For general questions about cloud services (e.g., 'What is EC2?', 'Explain Azure Functions'), always answer informatively if possible
      - The response should be formatted using Markdown syntax for better readability
      - Highlight important information and structure the answer for easy reading
      - When responding about cloud services, include accurate information about pricing, regions, and features if relevant
      - If making recommendations, consider best practices for cloud architecture and security
      - If the user is greeting you (saying hi, hello, etc.), respond with a warm, professional greeting that introduces yourself
      - When greeting, mention the time of day if appropriate, and express your readiness to help with their cloud needs
      
      Return JSON only in this format:
      {
        "answer": "Your complete helpful, conversational response to the user"
      }`;
    }

    // Generate conversation history for context
    const messagePayload = [
      new SystemMessage(promptText),
      ...this.generateConversationHistory(conversationHistory),
      new HumanMessage(String(query))
    ];

    console.log("Sending payload to LLM for unified response");
    try {
      console.log(`Processing query for unified response: "${query}"`);
      const llmResponse = await this.llm.invoke(messagePayload);
      
      try {
        const responseContent = String(llmResponse.content);
        console.log("Raw LLM response:", responseContent);
        
        let parsedResponse;
        try {
          // Try to parse the response as JSON
          parsedResponse = JSON.parse(responseContent);
          console.log("Parsed response:", parsedResponse);
          
          // For conversation history queries, handle structured responses
          if (isConversationHistoryQuery && 
              parsedResponse.role === 'assistant' && 
              parsedResponse.workflow === 'conversationSummary' &&
              parsedResponse.response) {
            
            // Ensure menu is present
            if (!parsedResponse.response.menu) {
              parsedResponse.response.menu = this.getMenuForCSP(csp || 'aws', true);
            }
            
            console.log("Using structured conversation summary response from LLM");
            return {
              ...cloudState,
              finalResponse: parsedResponse
            };
          }
          
          // If the parsed response already has the correct structure (role, workflow, response),
          // use it directly instead of rebuilding it
          if (parsedResponse.role === 'assistant' && 
              parsedResponse.workflow && 
              parsedResponse.response && 
              typeof parsedResponse.response.message === 'string') {
            
            // Just ensure the menu is updated
            if (!parsedResponse.response.menu) {
              parsedResponse.response.menu = this.getMenuForCSP(csp || 'aws', true);
            }
            
            console.log("Using structured response directly");
            return {
              ...cloudState,
              finalResponse: parsedResponse
            };
          }
        } catch (jsonError) {
          // If parsing fails, use the raw response as the message
          console.error("JSON parse error:", jsonError);
          parsedResponse = { answer: responseContent };
        }
        
        // Extract the response message, handling different structures
        let responseMessage: string;
        
        if (parsedResponse.answer) {
          // Standard expected format
          responseMessage = parsedResponse.answer;
        } else if (parsedResponse.response && parsedResponse.response.message) {
          // Handle case where response is nested in a response object
          responseMessage = parsedResponse.response.message;
        } else if (typeof parsedResponse === 'object') {
          // Try to find any string property that could be a message
          const stringProps = Object.entries(parsedResponse)
            .filter(([_, value]) => typeof value === 'string' && value.length > 0)
            .map(([_, value]) => value as string);
          
          if (stringProps.length > 0) {
            responseMessage = stringProps[0];
          } else {
            // If no string property is found, stringify the whole object
            responseMessage = JSON.stringify(parsedResponse);
          }
        } else {
          // Fallback to the raw response content
          responseMessage = responseContent;
        }
        
        // Create appropriate workflow type based on action
        let workflowType = "generalResponse";
        if (isConversationHistoryQuery) {
          workflowType = "conversationSummary";
        } else if (action) {
          switch (action.type) {
            case 'GENERAL_RESPONSE': workflowType = "generalResponse"; break;
            case 'DEPLOY': workflowType = "deployService"; break;
            case 'CONFIGURE': workflowType = "configureService"; break;
            case 'LIST_RESOURCES': workflowType = "listResources"; break;
            case 'VIEW_CSP_OPTIONS': workflowType = "viewCspOptions"; break;
            case 'SELECT_CSP': workflowType = "selectCsp"; break;
            default: workflowType = "generalResponse";
          }
        }
        
        // Extract details from the action payload or response
        let msgCsp = csp;
        let msgService = null;
        let msgRegion = null;
        let msgSpecs = {};
        
        if (action && action.payload) {
          msgCsp = action.payload.csp || msgCsp;
          msgService = action.payload.service || null;
          msgRegion = action.payload.region || null;
          msgSpecs = action.payload.specifications || {};
        }
        
        // For conversation history queries, extract the previous questions for reference
        let previousQuestions: string[] = [];
        if (isConversationHistoryQuery) {
          previousQuestions = conversationHistory
            .filter(msg => msg.role === 'human')
            .map(msg => typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content))
            .slice(0, -1); // Exclude the current question
        }
        
        // Create the final response object with the LLM's answer as the message
        return {
          ...cloudState,
          finalResponse: {
            role: "assistant",
            workflow: workflowType,
            response: {
              message: responseMessage, // Use the processed response message
              service: msgService,
              csp: msgCsp,
              region: msgRegion,
              specifications: msgSpecs,
              previousQuestions: isConversationHistoryQuery ? previousQuestions : undefined,
              menu: this.getMenuForCSP(msgCsp || 'aws', true)
            }
          }
        };
      } catch (error) {
        console.error("Error processing LLM response:", error);
        
        // Provide a fallback response when JSON parsing fails
        return this.createErrorResponse(cloudState, "I apologize, but I encountered an error processing your request. Could you please try again or rephrase your question?");
      }
    } catch (e) {
      console.error("Error in generateUnifiedResponse:", e);
      return this.createErrorResponse(cloudState, "I apologize, but I encountered an error processing your request. Could you please try again or rephrase your question?");
    }
  }
  
  // Helper method to create error responses
  private createErrorResponse(cloudState: CloudState, message: string): CloudState {
    return {
      ...cloudState,
      finalResponse: {
        role: "assistant",
        workflow: "errorResponse",
        response: {
          message: message,
          menu: this.getMenuForCSP(cloudState.csp || 'aws', true)
        }
      }
    };
  }

  // Helper: Generate menu for a given CSP
  private getMenuForCSP(csp: string, friendly = false): string[] {
    if (!this.servicesData) return ['No services data available.'];
    const cspLower = csp.toLowerCase();
    const services = this.servicesData.list.filter(s => s.cloud === cspLower);
    let menu: string[] = [];
    if (friendly) {
      menu = services.map(service => `Provision or manage a ${service.name}`);
      if (cspLower === 'aws') {
        menu.push('View AWS security groups');
        menu.push('Ask general questions about AWS');
      } else if (cspLower === 'azure') {
        menu.push('Ask general questions about Azure services');
        menu.push('Get help with Azure resources');
      } else if (cspLower === 'gcp') {
        menu.push('Ask general questions about Google Cloud');
        menu.push('Get help with GCP resources');
      } else if (cspLower === 'oracle') {
        menu.push('Ask general questions about Oracle Cloud');
        menu.push('Get help with Oracle Cloud resources');
      }
    } else {
      menu = services.map(service => `Provision a ${service.name}`);
      if (cspLower === 'aws') {
        menu.push('List the security groups');
        menu.push('Ask general questions about AWS');
      } else if (cspLower === 'azure') {
        menu.push('Ask general questions about Azure');
      } else if (cspLower === 'gcp') {
        menu.push('Ask general questions about GCP');
      } else if (cspLower === 'oracle') {
        menu.push('Ask general questions about Oracle Cloud');
      }
    }
    return menu;
  }

  // Helper: List AWS security groups
  private async listAWSSecurityGroups(): Promise<string> {
    // You may want to configure credentials/region here or use environment variables
    const ec2 = new AWS.EC2();
    try {
      const result = await ec2.describeSecurityGroups().promise();
      if (!result.SecurityGroups || result.SecurityGroups.length === 0) {
        return 'No security groups found.';
      }
      let output = 'Here are your AWS security groups:\n';
      result.SecurityGroups.forEach((sg, idx) => {
        output += `${idx + 1}. ${sg.GroupName} (ID: ${sg.GroupId})\n`;
      });
      return output;
    } catch (err) {
      return `Error retrieving security groups: ${err}`;
    }
  }

  // Helper: Load all conversations from file
  private loadAllConversationsFromFile(): Record<string, any> {
    try {
      if (conversationFs.existsSync(this.conversationFilePath)) {
        const content = conversationFs.readFileSync(this.conversationFilePath, 'utf8');
        return JSON.parse(content);
      }
    } catch (e) {
      console.error('Error loading conversation.json:', e);
    }
    return {};
  }

  // Helper: Save all conversations to file
  private saveAllConversationsToFile(conversations: Record<string, any>) {
    try {
      // Ensure the file exists before writing
      if (!conversationFs.existsSync(this.conversationFilePath)) {
        conversationFs.writeFileSync(this.conversationFilePath, '{}', 'utf8');
      }
      conversationFs.writeFileSync(this.conversationFilePath, JSON.stringify(conversations, null, 2), 'utf8');
    } catch (e) {
      console.error('Error saving conversations.json:', e);
    }
  }

  // Helper: Save/update a single user's conversation
  private saveUserConversation(userId: string, conversation: any) {
    let allConversations: Record<string, any> = {};
    try {
      allConversations = this.loadAllConversationsFromFile();
    } catch (e) {
      allConversations = {};
    }
    // Always overwrite the user's conversation with the in-memory version
    allConversations[userId] = conversation;
    this.saveAllConversationsToFile(allConversations);
  }

  private getISTTimestamp(): string {
    const now = new Date();
    return new Intl.DateTimeFormat('en-IN', {
      timeZone: 'Asia/Kolkata',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    }).format(now);
  }

  async processMessage(message: string, userId: string, csp?: string, fields?: any) {
    if (!userId) {
      throw new Error('User ID is required');
    }

    console.log("Processing message:", message);
    console.log("Fields:", fields);

    // --- Handle Deployment Requests with Form Data ---
    if (fields && fields.formData) {
      console.log("Detected form data for deployment");
      
      // Initialize or get user conversation
      if (!this.conversations.has(userId)) {
        this.conversations.set(userId, {
          csp: csp?.toLowerCase() || 'aws',
          history: []
        });
      }
      
      const userConversation = this.conversations.get(userId);
      userConversation.history.push({
        role: 'human',
        content: message,
        timestamp: this.getISTTimestamp()
      });

      // Get the service name from the action
      let serviceName = '';
      
      // Try to extract service name from message if not explicitly provided
      if (!serviceName) {
        const deployMatches = message.match(/deploy\s+(?:a|an)?\s*([a-zA-Z0-9\s]+)(?:\s+in|\s+on)?\s+([a-zA-Z0-9]+)?/i);
        if (deployMatches && deployMatches.length > 1) {
          serviceName = deployMatches[1].trim();
          // If no csp provided in fields but mentioned in message, use it
          if (!csp && deployMatches.length > 2 && deployMatches[2]) {
            csp = deployMatches[2].trim().toLowerCase();
          }
        }
      }
      
      // Use a default if we couldn't extract it
      if (!serviceName) {
        serviceName = "Virtual Machine"; // Default service name
      }

      const currentCSP = csp || userConversation.csp || 'aws';
      console.log(`Attempting to deploy ${serviceName} on ${currentCSP}`);

      // Look up the service config in our services.json
      const matchingService = this.findMatchingServiceByName(serviceName, currentCSP);
      
      let template = fields.template;
      // If a template wasn't provided, try to create one based on the service
      if (!template && matchingService) {
        try {
          template = this.generateTemplateForService(matchingService, fields.formData);
        } catch (templateError) {
          console.error("Error generating template:", templateError);
        }
      }

      // Call deployment tool
      const deploymentResult = await this.deployTool.deployService({
        serviceName: matchingService ? matchingService.name : serviceName,
        csp: currentCSP,
        userId,
        formData: fields.formData,
        template: template
      });

      // Create response object
      const responseObj = {
        role: 'assistant',
        workflow: 'deployment',
        response: {
          message: deploymentResult.success 
            ? `Deployment of ${serviceName} has been initiated successfully.`
            : `Failed to deploy ${serviceName}: ${deploymentResult.message}`,
          details: deploymentResult.details,
          deploymentId: deploymentResult.deploymentId,
          menu: this.getMenuForCSP(currentCSP, true)
        }
      };

      // Add response to conversation history
      userConversation.history.push({
        role: 'assistant',
        content: responseObj,
        timestamp: this.getISTTimestamp()
      });

      // Save updated conversation
      this.saveUserConversation(userId, userConversation);
      return responseObj;
    }

    // --- Determine CSP: message CSP > request body CSP > conversation CSP > default ---
    let userCSP = csp;
    // Check if message explicitly mentions a CSP
    const cspRegex = /(aws|azure|gcp|oracle)/i;
    const cspInMessage = message.match(cspRegex)?.[0]?.toLowerCase();
    if (cspInMessage) {
      userCSP = cspInMessage;
    } else if (csp) {
      userCSP = csp.toLowerCase();
    } else if (this.conversations.has(userId)) {
      userCSP = this.conversations.get(userId).csp || userCSP;
    } else {
      userCSP = 'aws';
    }

    // Ensure userCSP is always a string
    if (!userCSP) userCSP = 'aws';

    // Initialize or get user conversation
    if (!this.conversations.has(userId)) {
      this.conversations.set(userId, {
        csp: userCSP,
        history: []
      });
    }

    const userConversation = this.conversations.get(userId);
    userConversation.history.push({
      role: 'human',
      content: message,
      timestamp: this.getISTTimestamp()
    });

    // Pass the full conversation history to the cloudState for context-aware responses
    let cloudState: CloudState = {
      conversationHistory: userConversation.history,
      query: message,
      action: null,
      csp: userCSP,
      response: null
    };

    try {
      // Use the compiled workflow for processing - no more analyzeInput
      cloudState = await this.workflow.invoke(cloudState);
      
      // Get the final response
      const finalResponse = cloudState.finalResponse;
      
      // Always add menu to the response if it exists
      if (finalResponse && finalResponse.response) {
        finalResponse.response.menu = this.getMenuForCSP(cloudState.csp || 'aws', true);
      }
      
      // Add response to conversation history
      userConversation.history.push({
        role: 'assistant',
        content: finalResponse,
        timestamp: this.getISTTimestamp()
      });
      
      // After updating userConversation.history, persist to file
      this.saveUserConversation(userId, userConversation);
      
      return finalResponse;
    } catch (error) {
      console.error('Error processing message:', error);
      const errorResponse = {
        role: 'assistant',
        workflow: 'error',
        response: {
          message: 'Sorry, I encountered an error processing your request. Please try again.',
          menu: this.getMenuForCSP(userCSP || 'aws', true)
        }
      };
      
      userConversation.history.push({
        role: 'assistant',
        content: errorResponse,
        timestamp: this.getISTTimestamp()
      });
      
      this.saveUserConversation(userId, userConversation);
      return errorResponse;
    }
  }

  /**
   * Returns all user IDs that have conversation history.
   */
  getAllUserIds(): string[] {
    const allConversations = this.loadAllConversationsFromFile();
    return Object.keys(allConversations);
  }

  /**
   * Returns the conversation history for a specific userId, or null if not found.
   */
  getConversationByUserId(userId: string): any {
    const allConversations = this.loadAllConversationsFromFile();
    return allConversations[userId] || null;
  }

  // Helper method to generate a template based on service and form data
  private generateTemplateForService(service: ServiceConfig, formData: any): string {
    try {
      // This is a simplified template generator - expand based on your needs
      if (service.cloud.toLowerCase() === 'aws') {
        if (service.name.toLowerCase().includes('virtual machine') || 
            service.name.toLowerCase().includes('ec2')) {
          return `
{
  "AWSTemplateFormatVersion": "2010-09-09",
  "Resources": {
    "EC2Instance": {
      "Type": "AWS::EC2::Instance",
      "Properties": {
        "InstanceType": "${formData.instanceType || 't2.micro'}",
        "ImageId": "${formData.amiId || 'ami-0c55b159cbfafe1f0'}",
        "KeyName": "${formData.keyName || ''}",
        "Tags": [
          {
            "Key": "Name",
            "Value": "${formData.instanceName || 'EC2 Instance'}"
          }
        ]
      }
    }
  }
}`;
        } else if (service.name.toLowerCase().includes('s3') || 
                  service.name.toLowerCase().includes('bucket')) {
          return `
{
  "AWSTemplateFormatVersion": "2010-09-09",
  "Resources": {
    "S3Bucket": {
      "Type": "AWS::S3::Bucket",
      "Properties": {
        "BucketName": "${formData.bucketName || ''}",
        "AccessControl": "${formData.accessControl || 'Private'}"
      }
    }
  }
}`;
        }
      }
      
      // Return a basic template if we can't determine the service type
      return JSON.stringify({
        service: service.name,
        cloud: service.cloud,
        properties: formData
      }, null, 2);
      
    } catch (error) {
      console.error("Error generating template:", error);
      return JSON.stringify({ 
        error: "Failed to generate template",
        formData: formData
      });
    }
  }
} 