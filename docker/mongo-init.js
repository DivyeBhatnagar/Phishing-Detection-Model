// MongoDB Initialisation Script
// Runs once on first container startup
db = db.getSiblingDB('phishguard');

// Create application user
db.createUser({
  user: 'phishguard_app',
  pwd: 'phishguard_app_pass',
  roles: [{ role: 'readWrite', db: 'phishguard' }]
});

// Create collections with validation schemas
db.createCollection('predictions', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['email_hash', 'prediction', 'confidence', 'created_at'],
      properties: {
        prediction: { bsonType: 'string', enum: ['spam', 'legitimate'] },
        confidence: { bsonType: 'double', minimum: 0, maximum: 1 },
        risk_level: { bsonType: 'string', enum: ['low', 'medium', 'high'] }
      }
    }
  }
});

db.createCollection('users');
db.createCollection('logs');

// Create indexes
db.predictions.createIndex({ 'email_hash': 1 });
db.predictions.createIndex({ 'created_at': -1 });
db.predictions.createIndex({ 'prediction': 1 });
db.predictions.createIndex({ 'user_id': 1 });
db.predictions.createIndex({ 'risk_level': 1 });

db.users.createIndex({ 'username': 1 }, { unique: true });
db.users.createIndex({ 'email': 1 }, { unique: true });
db.users.createIndex({ 'api_key': 1 });

db.logs.createIndex({ 'created_at': -1 });
db.logs.createIndex({ 'endpoint': 1 });

print('PhishGuard MongoDB initialised successfully.');
