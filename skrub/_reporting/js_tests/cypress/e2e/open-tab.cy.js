describe('test open_tab parameter', () => {
    // Map each open_tab value to its corresponding tab/panel selectors and test file
    const tabs = {
        'sample': { tab: 'sample-tab', panel: 'sample-panel', file: 'employee_salaries.html' },
        'stats': { tab: 'summary-statistics-tab', panel: 'summary-statistics-panel', file: 'open_tab_stats.html' },
        'distributions': { tab: 'summaries-tab', panel: 'summaries-panel', file: 'open_tab_distributions.html' },
        'associations': { tab: 'associations-tab', panel: 'associations-panel', file: 'open_tab_associations.html' },
    };

    // Extract all tab and panel selectors for validation
    const allTabs = Object.values(tabs).map(t => t.tab);
    const allPanels = Object.values(tabs).map(t => t.panel);

    // Generate a test case for each open_tab value
    Object.entries(tabs).forEach(([name, { tab, panel, file }]) => {
        it(`opens on the ${name} tab when open_tab="${name}"`, () => {
            cy.visit(`_reports/${file}`);
            cy.get('skrub-table-report').shadow().as('report');

            // Verify the correct tab is selected and its panel is visible
            cy.get('@report').find(`[data-test="${tab}"]`).should('have.data', 'isSelected');
            cy.get('@report').find(`[data-test="${panel}"]`).should('be.visible');

            // Verify all other tabs are not selected
            allTabs.filter(t => t !== tab).forEach(t => {
                cy.get('@report').find(`[data-test="${t}"]`).should('not.have.data', 'isSelected');
            });

            // Verify all other panels are not visible
            allPanels.filter(p => p !== panel).forEach(p => {
                cy.get('@report').find(`[data-test="${p}"]`).should('not.be.visible');
            });
        });
    });
});
