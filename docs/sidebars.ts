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
      label: '🎯 General Users',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Getting Started',
          items: [
            'general-users/getting-started/installation',
            'general-users/getting-started/first-index',
            'general-users/getting-started/first-search',
          ],
        },
        {
          type: 'category',
          label: 'Basic Usage',
          items: [
            'general-users/basic-usage/basic-workflow',
            'general-users/basic-usage/document-types',
            'general-users/basic-usage/search-patterns',
          ],
        },
        {
          type: 'category',
          label: 'Use Cases',
          items: [
            'general-users/use-cases/technical-docs',
            'general-users/use-cases/meeting-notes',
            'general-users/use-cases/research-papers',
            'general-users/use-cases/personal-notes',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: '⚡ Power Users',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Integration & Automation',
          items: [
            'power-users/integration-automation/mcp-integration',
            'power-users/integration-automation/cli-workflows',
            'power-users/integration-automation/automation',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: '🔧 System Administrators',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Reference',
          items: [
            'system-administrators/reference/configuration',
          ],
        },
        {
          type: 'category',
          label: 'Troubleshooting',
          items: [
            'system-administrators/troubleshooting/troubleshooting',
          ],
        },
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