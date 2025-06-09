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
        'getting-started/installation',
        'getting-started/first-index',
        'getting-started/first-search',
      ],
    },
    {
      type: 'category',
      label: 'Usage Examples',
      items: [
        'usage-examples/basic-workflow',
        'usage-examples/document-types',
        'usage-examples/search-patterns',
      ],
    },
    {
      type: 'category',
      label: 'Real-world Scenarios',
      items: [
        'real-world-scenarios/technical-docs',
        'real-world-scenarios/meeting-notes',
        'real-world-scenarios/research-papers',
        'real-world-scenarios/personal-notes',
      ],
    },
    {
      type: 'category',
      label: 'Configuration & Optimization',
      items: [
        'configuration-optimization/configuration',
        'configuration-optimization/indexing-strategies',
        'configuration-optimization/search-optimization',
        'configuration-optimization/performance-tuning',
      ],
    },
    {
      type: 'category',
      label: 'Integration',
      items: [
        'integration/mcp-integration',
        'integration/cli-workflows',
        'integration/automation',
      ],
    },
    {
      type: 'category',
      label: 'Reference & Troubleshooting',
      items: [
        'reference-troubleshooting/troubleshooting',
        'reference-troubleshooting/cli-reference',
        'reference-troubleshooting/configuration-reference',
        'reference-troubleshooting/japanese-support',
      ],
    },
    {
      type: 'category',
      label: 'For Developers',
      items: [
        'for-developers/quickstart',
        'for-developers/architecture',
        'for-developers/cli',
        'for-developers/crawler',
        'for-developers/indexer',
        'for-developers/reranker',
        'for-developers/japanese',
        'for-developers/mcp_server',
        'for-developers/e2e_display_testing',
        'for-developers/installation-testing',
        'for-developers/installation-testing-readme',
        'for-developers/installation-troubleshooting',
        'for-developers/immutable-configuration-migration',
        'for-developers/release-process',
      ],
    },
  ],
};

export default sidebars;
