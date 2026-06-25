import test from 'node:test';
import assert from 'node:assert/strict';
import { claudeBody, openaiBody, DEFAULT_MODELS } from '../src/llm.js';

test('claudeBody is text-only (no audio attachment) with system prompt', () => {
  const b = claudeBody('claude-sonnet-4-6', 'SYS', 'USER');
  assert.equal(b.system, 'SYS');
  assert.equal(b.max_tokens, 1500);
  assert.deepEqual(b.messages, [{ role: 'user', content: 'USER' }]);
});

test('openaiBody puts system + user as plain text messages', () => {
  const b = openaiBody('gpt-4o', 'SYS', 'USER');
  assert.deepEqual(b.messages, [
    { role: 'system', content: 'SYS' },
    { role: 'user', content: 'USER' },
  ]);
});

test('default models match the Python CLI defaults', () => {
  assert.equal(DEFAULT_MODELS.claude, 'claude-sonnet-4-6');
  assert.equal(DEFAULT_MODELS.openai, 'gpt-4o');
});
