describe ('test the columns tab', () => {
    it ('displays the selected columns', () => {
        cy.visit('_reports/employee_salaries.html');
        cy.get('skrub-table-report').shadow().as('report');
        cy.get('@report').find('button[data-target-tab="columns-tab"]').click();
        cy.get('@report').find('#deselect-all-cols').click();
        cy.get('@report').find('#selected-cols-box').as('selectedCols').should('have.text', '[]');
        cy.get('@report').find('#col_1').find('.select-column-checkbox').check();
        cy.get('@selectedCols').should('have.text', "['department']");
    });
});
