export interface ServiceConfig {
  id: string;
  name: string;
  template: string;
  price: number;
  cloud: string;
  available: boolean;
  requiredFields: Array<{
    type: string;
    fieldId: string;
    fieldName: string;
    fieldValue: string;
    fieldTypeValue: string;
    dependent: boolean;
    dependentON: string;
    dependentFOR: string;
  }>;
}

export interface ServicesData {
  list: ServiceConfig[];
} 