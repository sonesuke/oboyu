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
      collapsed: false,
      items: [
        'getting-started/installation',
        'getting-started/first-index',
        'getting-started/first-search',
      ],
    },
    {
      type: 'category',
      label: 'Basic Usage',
      collapsed: false,
      items: [
        'basic-usage/basic-workflow',
        'basic-usage/document-types',
        'basic-usage/search-patterns',
      ],
    },
    {
      type: 'category',
      label: 'Use Cases',
      collapsed: false,
      items: [
        'use-cases/csv-enrichment',
        'use-cases/technical-docs',
        'use-cases/research-papers',
        'use-cases/personal-notes',
        'use-cases/nikkei225-securities-reports',
        'use-cases/github-issues-search',
      ],
    },
    {
      type: 'category',
      label: 'Integration & Automation',
      collapsed: false,
      items: [
        'integration-automation/mcp-integration',
        'integration-automation/cli-workflows',
        'integration-automation/automation',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      collapsed: false,
      items: [
        'reference/configuration',
      ],
    },
    {
      type: 'category',
      label: 'Troubleshooting',
      collapsed: false,
      items: [
        'troubleshooting/troubleshooting',
      ],
    },
    {
      type: 'category',
      label: '💻 Developers/Contributors',
      collapsed: true,
      items: [
        {
          type: 'category',
          label: 'Architecture',
          items: [
            'for-developers/architecture',
            'for-developers/architecture/query-engine',
          ],
        },
        {
          type: 'category',
          label: 'Core Components',
          items: [
            'for-developers/crawler',
            'for-developers/indexer',
            'for-developers/mcp_server',
          ],
        },
        {
          type: 'category',
          label: 'Development',
          items: [
            'for-developers/quickstart',
            'for-developers/cli',
            'for-developers/reranker',
            'for-developers/japanese',
            'for-developers/e2e_display_testing',
            'for-developers/pre-commit-optimization',
            'for-developers/ci-cd-optimization',
            'for-developers/release-process',
          ],
        },
        {
          type: 'category',
          label: 'Testing & Installation',
          items: [
            'for-developers/installation-testing',
            'for-developers/installation-testing-readme',
            'for-developers/installation-troubleshooting',
            'for-developers/immutable-configuration-migration',
          ],
        },
      ],
    },
  ],
};

export default sidebars;