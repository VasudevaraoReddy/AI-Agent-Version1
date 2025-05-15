import { Injectable } from '@nestjs/common';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { StateGraph, START, END } from '@langchain/langgraph';
import * as fs from 'fs';
import * as path from 'path';
import * as AWS from 'aws-sdk';
import * as conversationFs from 'fs';
import { z } from 'zod';

interface CloudState {
  conversationHistory: any[];
  query: string;
  action: any;
  csp: string | null;
  response: any;
  analysis?: any;
  finalResponse?: any;
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
  private readonly genAI: GoogleGenerativeAI;
  private conversations: Map<string, any> = new Map();
  private servicesData: ServicesData | null = null;
  private conversationFilePath = path.join(process.cwd(), 'conversations.json');

  constructor() {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error('GEMINI_API_KEY environment variable is not set');
    }
    this.genAI = new GoogleGenerativeAI(apiKey);
    this.loadServicesData();

    // Load all conversations from file into memory
    const allConversations = this.loadAllConversationsFromFile();
    Object.entries(allConversations).forEach(([userId, convo]) => {
      this.conversations.set(userId, convo);
    });

    // LangGraph workflow setup
    const CloudStateSchema = z.object({
      conversationHistory: z.array(z.any()),
      query: z.string(),
      action: z.any(),
      csp: z.string().nullable(),
      response: z.any(),
      analysis: z.any().nullable(),
      finalResponse: z.any().nullable(),
    });
    const workflow = new StateGraph(CloudStateSchema)
      .addNode('analyzeInput', async (state: CloudState) => await this.analyzeInput(state.query, state))
      .addNode('findAction', async (state: CloudState) => await this.findAction(state))
      .addNode('processAction', async (state: CloudState) => await this.processAction(state))
      .addNode('generateUnifiedResponse', async (state: CloudState) => await this.generateUnifiedResponse(state));
    workflow.addEdge(START, 'analyzeInput');
    workflow.addEdge('analyzeInput', 'findAction');
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

  private async analyzeInput(userInput: string, cloudState: CloudState): Promise<CloudState> {
    const prompt = `You analyze cloud-related queries to determine if they are general questions or action requests.
    
    Return JSON only:
    {
      "type": "general" | "action",
      "csp": "aws" | "azure" | "gcp" | null,
      "actionType": "deploy" | "provision" | "configure" | null,
      "details": {
        "service": string | null,
        "region": string | null,
        "specifications": object | null
      }
    }

    Rules:
    - If the query is about general cloud concepts, set type to "general"
    - If the query contains action words like deploy, provision, spin up, set up, configure, build, instantiate, generate, initialize, launch, install, setup, fabricate, construct, activate, start, run, push, release, roll out, go live, make live, bring online, distribute, execute, publish, kick off, open, provision, install, set type to "action"
    - Extract cloud provider (AWS, Azure, GCP) if mentioned
    - For action type, extract relevant service, region, and specifications
    - Leave fields as null if not present in the query
    - Return ONLY the JSON object, no markdown formatting or additional text`;

    const model = this.genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });
    const result = await model.generateContent([prompt, userInput]);
    const response = result.response.text();

    try {
      const cleanedResponse = this.cleanJsonResponse(response);
      const parsedResult = JSON.parse(cleanedResponse);
      // Ensure csp is set if missing
      if (!parsedResult.csp && cloudState.csp) {
        parsedResult.csp = cloudState.csp;
      }
      return {
        ...cloudState,
        query: userInput,
        analysis: parsedResult
      };
    } catch (e) {
      console.error("Error parsing analyzeInput result:", e);
      return {
        ...cloudState,
        query: userInput,
        analysis: {
          type: "general",
          csp: null,
          actionType: null,
          details: null
        }
      };
    }
  }

  private async findAction(cloudState: CloudState): Promise<CloudState> {
    const { analysis } = cloudState;

    if (analysis.type === "general") {
      // For general questions, generate an informative response
      const prompt = `You are a cloud expert. Provide a clear and informative response to the following question about cloud services.
      
      Question: ${cloudState.query}
      
      Return JSON only:
      {
        "role": "assistant",
        "workflow": "generalResponse",
        "response": {
          "message": string,
          "details": {
            "explanation": string,
            "keyPoints": string[],
            "relatedServices": string[]
          }
        }
      }`;

      const model = this.genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });
      const result = await model.generateContent([prompt]);
      const response = result.response.text();

      try {
        const cleanedResponse = this.cleanJsonResponse(response);
        const parsedResult = JSON.parse(cleanedResponse);
        return {
          ...cloudState,
          finalResponse: parsedResult
        };
      } catch (e) {
        console.error("Error parsing general response:", e);
        return {
          ...cloudState,
          finalResponse: {
            role: "assistant",
            workflow: "generalResponse",
            response: {
              message: "I apologize, but I encountered an error processing your question.",
              details: {
                explanation: "",
                keyPoints: [],
                relatedServices: []
              }
            }
          }
        };
      }
    }

    // For action requests, proceed with normal action processing
    const prompt = `Based on the analysis, determine the appropriate action.
    
    Return JSON only:
    {
      "action": {
        "type": "GENERAL_RESPONSE" | "DEPLOY" | "PROVISION" | "CONFIGURE",
        "payload": {
          "service": string,
          "region": string,
          "specifications": object,
          "csp": string
        }
      }
    }`;

    const model = this.genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });
    const result = await model.generateContent([prompt, JSON.stringify(analysis)]);
    const response = result.response.text();

    try {
      const cleanedResponse = this.cleanJsonResponse(response);
      const parsedResult = JSON.parse(cleanedResponse);
      return {
        ...cloudState,
        action: parsedResult.action
      };
    } catch (e) {
      console.error("Error parsing findAction result:", e);
      return {
        ...cloudState,
        action: {
          type: "GENERAL_RESPONSE",
          payload: {}
        }
      };
    }
  }

  private async processAction(cloudState: CloudState): Promise<CloudState> {
    const { action, analysis } = cloudState;

    if (cloudState.finalResponse) {
      // If we already have a final response (from general question), skip action processing
      return cloudState;
    }

    // Check if this is a deploy/provision/create action
    if (analysis && analysis.type === 'action') {
      const matchingService = this.findMatchingService(analysis);
      if (matchingService) {
        // Return the service configuration with required fields
        const response = {
          status: "service_found",
          message: `Found service configuration for ${matchingService.name} on ${matchingService.cloud.toUpperCase()}`,
          service: {
            name: matchingService.name,
            description: matchingService.description,
            cloud: matchingService.cloud,
            requiredFields: matchingService.requiredFields
          },
          details: analysis.details
        };

        return {
          ...cloudState,
          response,
          finalResponse: {
            role: "assistant",
            workflow: "serviceConfiguration",
            response: {
              message: `To provision ${matchingService.name} on ${matchingService.cloud.toUpperCase()}, please provide the following information:`,
              service: matchingService,
              details: analysis.details
            }
          }
        };
      } else {
        // Service not found: reply politely and show available services for the CSP
        const csp = (analysis && analysis.csp) ? analysis.csp : 'azure';
        const availableServices = this.servicesData?.list.filter(s => s.cloud === csp.toLowerCase()).map(s => s.name) || [];
        return {
          ...cloudState,
          finalResponse: {
            role: "assistant",
            workflow: "serviceNotFound",
            response: {
              message: `Sorry, the selected service is currently not available to provision in ${csp.toUpperCase()}. Please select from the available services below:`,
              availableServices,
              menu: this.getMenuForCSP(csp, true)
            }
          }
        };
      }
    }

    // Fall back to original behavior if no matching service found
    const response = {
      status: "success",
      message: `Processed ${action.type} action for ${action.payload.csp}`,
      details: action.payload
    };

    return {
      ...cloudState,
      response
    };
  }

  private async generateUnifiedResponse(cloudState: CloudState): Promise<CloudState> {
    const { action, response, finalResponse } = cloudState;

    if (finalResponse) {
      // If we already have a final response, return it
      return cloudState;
    }

    // Check if we have a service configuration response
    if (response && response.status === 'service_found') {
      return {
        ...cloudState,
        finalResponse: {
          role: "assistant",
          workflow: "serviceConfiguration",
          response: {
            message: response.message,
            service: response.service,
            details: response.details
          }
        }
      };
    }

    // Original unified response generation
    const prompt = `Generate a user-friendly response based on the action and response.
    
    Return JSON only:
    {
      "role": "assistant",
      "workflow": "generateUnifiedResponse",
      "response": {
        "message": string,
        "details": object
      }
    }`;

    const model = this.genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });
    const result = await model.generateContent([prompt, JSON.stringify({ action, response })]);
    const aiResponse = result.response.text();

    try {
      const cleanedResponse = this.cleanJsonResponse(aiResponse);
      const parsedResult = JSON.parse(cleanedResponse);
      return {
        ...cloudState,
        finalResponse: parsedResult
      };
    } catch (e) {
      console.error("Error parsing generateUnifiedResponse result:", e);
      return {
        ...cloudState,
        finalResponse: {
          role: "assistant",
          workflow: "generateUnifiedResponse",
          response: {
            message: "I apologize, but I encountered an error processing your request.",
            details: {}
          }
        }
      };
    }
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

  async processMessage(message: string, userId: string, csp?: string) {
    if (!userId) {
      throw new Error('User ID is required');
    }

    // --- Handle Greetings ---
    const greetingRegex = /^(hi|hello|hey|greetings|good (morning|afternoon|evening)|howdy|sup|yo|hi there|hello there)$/i;
    if (greetingRegex.test(message.trim())) {
      // Initialize or get user conversation
      if (!this.conversations.has(userId)) {
        this.conversations.set(userId, {
          csp: csp?.toLowerCase() || 'aws',
          history: []
        });
      }
      const userConversation = this.conversations.get(userId);
      userConversation.history.push(['human', message]);

      // Generate greeting response using LLM
      const prompt = `You are a professional cloud services assistant. Generate a warm, professional greeting response.
      
      Current time: ${new Date().toLocaleTimeString()}
      User's Cloud Provider: ${userConversation.csp?.toUpperCase() || 'AWS'}
      
      Return JSON only:
      {
        "role": "assistant",
        "workflow": "greeting",
        "response": {
          "message": string, // A professional greeting that considers time of day and CSP
          "menu": array // Will be populated with the CSP menu
        }
      }

      Guidelines:
      - Be professional but warm
      - Consider the time of day
      - Mention the user's cloud provider if known
      - Keep the message concise and welcoming
      - Focus on being helpful with cloud services`;

      const model = this.genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });
      const result = await model.generateContent([prompt]);
      const response = result.response.text();

      try {
        const cleanedResponse = this.cleanJsonResponse(response);
        const parsedResult = JSON.parse(cleanedResponse);
        const responseObj = {
          ...parsedResult,
          response: {
            ...parsedResult.response,
            menu: this.getMenuForCSP(userConversation.csp || 'aws', true)
          }
        };
        userConversation.history.push(['assistant', responseObj]);
        this.saveUserConversation(userId, userConversation);
        return responseObj;
      } catch (e) {
        console.error("Error parsing greeting response:", e);
        // Fallback response if LLM fails
        const fallbackResponse = {
          role: 'assistant',
          workflow: 'greeting',
          response: {
            message: `Hello! I'm your cloud services assistant. How can I help you today?`,
            menu: this.getMenuForCSP(userConversation.csp || 'aws', true)
          }
        };
        userConversation.history.push(['assistant', fallbackResponse]);
        this.saveUserConversation(userId, userConversation);
        return fallbackResponse;
      }
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

    // --- Dynamic CSP Selection ---
    const cspIntentRegex = /(select|choose|go with|opt|prefer|use|pick|switch to|set|change to|work with|move to|start with|begin with|try|test|explore|want to go with|will opt|will select|will choose|will go with|will use|will pick|will switch to|will set|will change to|will work with|will move to|will start with|will begin with|will try|will test|will explore)/i;
    let selectedCSP: string | null = null;
    if (cspRegex.test(message) && cspIntentRegex.test(message)) {
      selectedCSP = message.match(cspRegex)?.[0].toLowerCase() || null;
    }
    if (selectedCSP) {
      // Save CSP in conversation
      if (!this.conversations.has(userId)) {
        this.conversations.set(userId, { csp: selectedCSP, history: [] });
      } else {
        this.conversations.get(userId).csp = selectedCSP;
      }
      // Append to conversation history
      const userConversation = this.conversations.get(userId);
      userConversation.history.push(['human', message]);
      const responseObj = {
        role: 'assistant',
        workflow: 'cspSelection',
        response: {
          message: `You have selected **${selectedCSP.toUpperCase()}** as your Cloud Service Provider.`,
          menu: this.getMenuForCSP(selectedCSP || 'aws', true)
        }
      };
      userConversation.history.push(['assistant', responseObj]);
      this.saveUserConversation(userId, userConversation);
      return responseObj;
    }

    // --- General Questions Mode ---
    if (/ask general questions|general questions|just chat|let's chat|i want to chat|i want to ask|i want to talk|i want to discuss|i want to know|i want to learn|i want to understand|i want to explore/i.test(message.trim())) {
      // Enter general chat mode for the user's CSP
      if (!this.conversations.has(userId)) {
        this.conversations.set(userId, { csp: userCSP, history: [] });
      }
      this.conversations.get(userId).generalChatMode = true;
      // Append to conversation history
      const userConversation = this.conversations.get(userId);
      userConversation.history.push(['human', message]);
      const responseObj = {
        role: 'assistant',
        workflow: 'generalChatMode',
        response: {
          message: `You are now in general chat mode for ${userCSP.toUpperCase()}. Ask me any cloud-related question!`,
          menu: this.getMenuForCSP(userCSP || 'aws', true)
        }
      };
      userConversation.history.push(['assistant', responseObj]);
      this.saveUserConversation(userId, userConversation);
      return responseObj;
    }

    // --- New Flow: List commands ---
    if (/^list/i.test(message.trim()) && userCSP.toLowerCase() === 'aws') {
      if (/security groups?/i.test(message)) {
        const sgList = await this.listAWSSecurityGroups();
        return {
          role: 'assistant',
          workflow: 'awsListSecurityGroups',
          response: {
            message: sgList + '\n\n' + this.getMenuForCSP('aws')
          }
        };
      }
      // Add more AWS list commands here if needed
    }

    // Initialize or get user conversation
    if (!this.conversations.has(userId)) {
      this.conversations.set(userId, {
        csp: userCSP,
        history: []
      });
    }

    const userConversation = this.conversations.get(userId);
    userConversation.history.push(['human', message]);

    // Pass the full conversation history to the cloudState for context-aware responses
    let cloudState: CloudState = {
      conversationHistory: userConversation.history,
      query: message,
      action: null,
      csp: userCSP,
      response: null
    };

    try {
      // Use the compiled workflow for processing
      cloudState = await this.workflow.invoke(cloudState);
      // For general questions, only answer if cloud-related, else respond professionally
      if (cloudState.finalResponse && cloudState.finalResponse.workflow === 'generalResponse') {
        const cloudKeywords = /cloud|aws|azure|gcp|oracle|virtual machine|load balancer|database|resource group|infrastructure|compute|storage|network|devops|cloud service|cloud provider|cloud platform|cloud resource|cloud deployment|cloud security|cloud cost|cloud billing|cloud automation|cloud api|cloud sdk|cloud cli|cloud shell|cloud console|cloud region|cloud zone|cloud subscription|cloud tenant|cloud identity|cloud permission|cloud policy|cloud quota|cloud monitoring|cloud logging|cloud alert|cloud backup|cloud disaster|cloud failover|cloud migration|cloud hybrid|cloud multi|cloud native|cloud app|cloud function|cloud container|cloud kubernetes|cloud docker|cloud serverless|cloud paas|cloud iaas|cloud saas|cloud ml|cloud ai|cloud analytics|cloud big data|cloud sql|cloud nosql|cloud dns|cloud cdn|cloud firewall|cloud vpn|cloud vpc|cloud subnet|cloud peering|cloud direct connect|cloud expressroute|cloud interconnect|cloud api gateway|cloud load balancer|cloud autoscale|cloud scaling|cloud elasticity|cloud orchestration|cloud pipeline|cloud workflow|cloud automation|cloud deployment|cloud provisioning|cloud template|cloud formation|cloud arm|cloud bicep|cloud terraform|cloud pulumi|cloud ansible|cloud chef|cloud puppet|cloud salt|cloud monitoring|cloud logging|cloud tracing|cloud metrics|cloud dashboard|cloud insights|cloud advisor|cloud recommender|cloud security|cloud compliance|cloud encryption|cloud key|cloud vault|cloud secret|cloud identity|cloud access|cloud sso|cloud iam|cloud ad|cloud directory|cloud user|cloud group|cloud role|cloud policy|cloud permission|cloud audit|cloud billing|cloud cost|cloud budget|cloud invoice|cloud charge|cloud spend|cloud usage|cloud quota|cloud limit|cloud cap|cloud alert|cloud notification|cloud event|cloud incident|cloud ticket|cloud support|cloud help|cloud docs|cloud documentation|cloud guide|cloud tutorial|cloud example|cloud sample|cloud reference|cloud api|cloud sdk|cloud cli|cloud shell|cloud console|cloud portal|cloud dashboard|cloud ui|cloud ux|cloud experience|cloud feedback|cloud survey|cloud review|cloud rating|cloud score|cloud benchmark|cloud test|cloud trial|cloud free|cloud credit|cloud offer|cloud promo|cloud discount|cloud coupon|cloud deal|cloud marketplace|cloud partner|cloud reseller|cloud distributor|cloud vendor|cloud provider|cloud platform|cloud service|cloud solution|cloud product|cloud feature|cloud capability|cloud function|cloud operation|cloud task|cloud job|cloud run|cloud execute|cloud invoke|cloud trigger|cloud event|cloud schedule|cloud cron|cloud timer|cloud batch|cloud workflow|cloud pipeline|cloud automation|cloud deployment|cloud provisioning|cloud template|cloud formation|cloud arm|cloud bicep|cloud terraform|cloud pulumi|cloud ansible|cloud chef|cloud puppet|cloud salt/gi;
        if (!cloudKeywords.test(message)) {
          // Not cloud-related, respond professionally
          const professionalResponse = {
            role: 'assistant',
            workflow: 'nonCloudQuestion',
            response: {
              message: 'I am designed to assist with cloud-related topics and queries. Please ask a question related to cloud computing, cloud services, or cloud platforms.'
            }
          };
          userConversation.history.push(['assistant', professionalResponse]);
          this.saveUserConversation(userId, userConversation);
          return professionalResponse;
        }
      }

      const finalResponse = cloudState.finalResponse;
      // Always add menu to the response
      let cspForMenu = cloudState.csp || (typeof csp === 'string' ? csp : undefined);
      if (finalResponse && finalResponse.response) {
        finalResponse.response.menu = this.getMenuForCSP(cspForMenu || 'aws');
      }
      userConversation.history.push(['assistant', finalResponse]);
      // After updating userConversation.history, persist to file
      this.saveUserConversation(userId, this.conversations.get(userId));
      return finalResponse;
    } catch (error) {
      console.error('Error processing message:', error);
      throw new Error('Failed to process the query with the language model.');
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
} 