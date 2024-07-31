describe('test the column summaries', () => {
    it('displays the selected columns', () => {
        cy.get('@report').find(
            'button[data-target-panel-id="column-summaries-panel"]'
        ).click();
        cy.get('@report').find('[data-test="deselect-all-columns"]')
            .click();
        cy.get('@report').find('#selected-columns-display').as(
            'selectedColumns').should('have.text', '[]');
        cy.get('@report').find('#col_1').find(
            '[data-role="select-column-checkbox"]').check();
        cy.get('@selectedColumns').should('have.text',
            "['department']");

        cy.get('@report').find('[data-test="column-filter-select"]')
            .select('Numeric columns');
        cy.get('@selectedColumns').should('have.text',
            "['department']");
        cy.get('@report').find('[data-test="select-all-columns"]')
            .click();
        cy.get('@selectedColumns').should('have.text',
            "['year_first_hired']");
    });
});
