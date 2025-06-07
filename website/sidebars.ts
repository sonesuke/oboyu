import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  mainSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: 'Introduction',
    },
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'quickstart',
        'cli',
        'configuration',
      ],
    },
    {
      type: 'category',
      label: 'Core Features',
      items: [
        'indexer',
        'query-engine',
        'crawler',
        'reranker',
      ],
    },
    {
      type: 'category',
      label: 'Advanced Topics',
      items: [
        'architecture',
        'japanese',
        'mcp-server',
        'immutable-configuration-migration',
      ],
    },
    {
      type: 'category',
      label: 'Help & Support',
      items: [
        'troubleshooting',
      ],
    },
  ],
};

export default sidebars;
