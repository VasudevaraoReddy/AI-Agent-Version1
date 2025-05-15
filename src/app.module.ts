import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { AgentModule } from './agents/agent.module';

@Module({
  imports: [AgentModule],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
