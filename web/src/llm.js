// Browser-direct model call. The analysis is text-only (no audio is ever sent),
// mirroring music_describer: build_user_prompt feeds the model the analysis JSON.
// The user's key lives only in memory and goes only to the chosen provider.

export const DEFAULT_MODELS = { claude: 'claude-sonnet-4-6', openai: 'gpt-4o' };

export function claudeBody(model, systemPrompt, userPrompt) {
  return {
    model,
    max_tokens: 1500,
    system: systemPrompt,
    messages: [{ role: 'user', content: userPrompt }],
  };
}

export function openaiBody(model, systemPrompt, userPrompt) {
  return {
    model,
    max_tokens: 1500,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt },
    ],
  };
}

export async function describe({ provider, model, key, systemPrompt, userPrompt }) {
  if (provider === 'claude') {
    const r = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'x-api-key': key,
        'anthropic-version': '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true',
        'content-type': 'application/json',
      },
      body: JSON.stringify(claudeBody(model, systemPrompt, userPrompt)),
    });
    if (!r.ok) throw new Error(`Claude ${r.status}: ${await r.text()}`);
    return (await r.json()).content[0].text;
  }
  const r = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { Authorization: `Bearer ${key}`, 'content-type': 'application/json' },
    body: JSON.stringify(openaiBody(model, systemPrompt, userPrompt)),
  });
  if (!r.ok) throw new Error(`OpenAI ${r.status}: ${await r.text()}`);
  return (await r.json()).choices[0].message.content;
}
