import { Controller, Post, Body, Get, Param } from '@nestjs/common';
import { AgentService } from './agent.service';

interface ChatRequestBody {
  message: string;
  userId: string;
  csp?: string;
  fields?: {
    formData: Record<string, string>;
    template: string;
  };
}

@Controller('agent')
export class AgentController {
  constructor(private readonly agentService: AgentService) {}

  @Post('chat')
  async chat(@Body() body: ChatRequestBody) {
    return this.agentService.processMessage(body.message, body.userId, body.csp, body.fields);
  }

  @Get('conversations')
  getAllUserIds() {
    return this.agentService.getAllUserIds();
  }

  @Get('conversations/:userId')
  getConversationByUserId(@Param('userId') userId: string) {
    return this.agentService.getConversationByUserId(userId);
  }
} 