describe('test using the copybuttons', {
    browser: 'electron'
}, () => {
    it('copies target content to clipboard', () => {
        cy.get('@report').find('button[data-target-panel-id="column-summaries-panel"]')
            .click();
        cy.get('@report').find('#col_1').as('col1').find(
            '[data-test="frequent-values-details"]').click();
        cy.get('@col1').find('[data-test="frequent-value-1"]').find("button").click({
            force: true
        });
        cy.window().its('navigator.clipboard').then((clip) => clip
            .readText()).should('be.equal', "'HHS'");
    });
});
