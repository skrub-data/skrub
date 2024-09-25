describe('test filtering visible columns', () => {
    it('hides columns not matched by the selector', () => {
        cy.get('@report').find('[data-test="n-columns-display"]').as(
            'nColumns').should('have.text', '8');

        cy.get('@report').find('[data-test="summaries-tab"]').as(
            'summariesTab').click();
        cy.get('@report').find('[data-test="column-filter-select"]')
            .select('Numeric columns');
        cy.get('@report').find('#col_7').should('be.visible');
        cy.get('@report').find('#col_0').should('not.be.visible');
        cy.get('@report').find('[data-test="sample-tab"]').as(
            'sampleTab').click();
        cy.get('@report').find('[data-role="dataframe-data"][data-i="1"][data-j="7"]').as(
            'cell7').should('be.visible');
        cy.get('@report').find('[data-role="dataframe-data"][data-i="1"][data-j="0"]').as(
            'cell0').should('not.be.visible');
        cy.get('@nColumns').should('have.text', '1');

        cy.get('@report').find('[data-test="column-filter-select"]')
            .select('Non-numeric columns');
        cy.get('@cell7').should('not.be.visible');
        cy.get('@cell0').should('be.visible');
        cy.get('@nColumns').should('have.text', '7');
        cy.get('@summariesTab').click();
        cy.get('@report').find('#col_7').should('not.be.visible');
        cy.get('@report').find('#col_0').should('be.visible');

        cy.get('@report').find('[data-test="column-filter-select"]')
            .select('Datetime columns');
        cy.get('@report').find('[data-test="summaries-panel"]').find(
            '[data-test="show-all-columns-button"]').should(
            'be.visible');
        cy.get('@sampleTab').click();
        cy.get('@nColumns').should('not.be.visible');
        cy.get('@report').find('[data-test="sample-panel"]').find(
            '[data-test="show-all-columns-button"]').click();
        cy.get('@nColumns').should('have.text', '8');
    });

    it('only shows the select input in columns and sample tabs', () => {
        cy.get('@report').find('[data-test="column-filter-select"]').as(
            'select');
        cy.get('@select').should('be.visible');
        cy.get('@report').find(
            'button[data-target-panel-id="column-summaries-panel"]'
        ).click();
        cy.get('@select').should('be.visible');
        cy.get('@report').find(
            'button[data-target-panel-id="column-associations-panel"]'
        ).click();
        cy.get('@select').should('not.be.visible');
        cy.get('@report').find(
            'button[data-target-panel-id="column-summaries-panel"]'
        ).click();
        cy.get('@select').should('be.visible');
    });
});
