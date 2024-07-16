describe ('test filtering visible columns', () => {
    it('hides columns not matched by the selector', () => {
        cy.visit('_reports/employee_salaries.html');
        cy.get('skrub-table-report').shadow().as('report');
        cy.get('@report').find('button[data-target-tab="columns-tab"]').click();
        cy.get('@report').find('#col-filter-select').select('Numeric columns');
        cy.get('@report').find('#col_7').should('be.visible');
        cy.get('@report').find('#col_0').should('not.be.visible');
        cy.get('@report').find('#col-filter-select').select('Non-numeric columns');
        cy.get('@report').find('#col_7').should('not.be.visible');
        cy.get('@report').find('#col_0').should('be.visible');
    });
});
