describe('test using the copybuttons', {
    browser: 'electron'
}, () => {
    it('copies target content to clipboard', () => {
        cy.visit('_reports/employee_salaries.html');
        cy.get('skrub-table-report').shadow().as('report');
        cy.get('@report').find('button[data-target-tab="columns-tab"]')
            .click();
        cy.get('@report').find('#col_1').as('col1').find(
            '[data-test="frequent-values-details"]').click();
        cy.get('@col1').find(
            '[data-test="frequent-values-select-snippet"]').select(
            "repr");
        cy.get('@col1').find('[data-test="copybutton-1"]').click({
            force: true
        });
        cy.window().its('navigator.clipboard').then((clip) => clip
            .readText()).should('be.equal', "'HHS'");

    });
});
