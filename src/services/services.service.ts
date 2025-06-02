import { Injectable } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';
import { ServiceConfig, ServicesData } from './types';

@Injectable()
export class ServicesService {
  private servicesData: ServicesData | null = null;

  constructor() {
    this.loadServicesData();
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

  getAllServices(): ServiceConfig[] {
    return this.servicesData?.list || [];
  }

  getServicesByFilter(name?: string, cloud?: string): ServiceConfig[] {
    if (!this.servicesData) return [];

    return this.servicesData.list.filter((service) => {
      const nameMatch = !name || service.name.toLowerCase().includes(name.toLowerCase());
      const cloudMatch = !cloud || service.cloud.toLowerCase() === cloud.toLowerCase();
      return nameMatch && cloudMatch;
    });
  }
} 