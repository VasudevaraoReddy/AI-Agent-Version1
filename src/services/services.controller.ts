import { Controller, Get, Query } from '@nestjs/common';
import { ServicesService } from './services.service';
import { ServiceConfig } from './types';

@Controller('services')
export class ServicesController {
  constructor(private readonly servicesService: ServicesService) {}

  @Get()
  getAllServices(): ServiceConfig[] {
    return this.servicesService.getAllServices();
  }

  @Get('filter')
  getServicesByFilter(
    @Query('name') name?: string,
    @Query('cloud') cloud?: string,
  ): ServiceConfig[] {
    return this.servicesService.getServicesByFilter(name, cloud);
  }
} 