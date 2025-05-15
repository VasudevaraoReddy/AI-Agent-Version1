export interface CloudResourceField {
  type: 'input';
  fieldId: string;
  fieldName: string;
  fieldValue: string;
  fieldTypeValue: string;
}

export interface CloudResourceSchema {
  name: string;
  description: string;
  cloud: 'aws' | 'azure' | 'gcp';
  requiredFields: CloudResourceField[];
} 