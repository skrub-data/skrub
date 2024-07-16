describe('test the dataframe sample tab', () => {
  it('shows a column card when clicking a column and can close it', () => {
      cy.visit('_reports/employee_salaries.html');
      cy.get('skrub-table-report').shadow().as('report');
      cy.get('@report').find('td[data-column-idx="1"]').first().click();
      cy.get('@report').find('#col_1_in_sample_tab').as('col1Card').should('be.visible');
      cy.get('@report').find('td[data-column-idx="2"]').first().click();
      cy.get('@report').find('#col_1_in_sample_tab').should('not.be.visible');
      cy.get('@report').find('#col_2_in_sample_tab').as('col2Card').should('be.visible');
      cy.get('@col2Card').find('.close-card-button').click({force: true});
      cy.get('@col2Card').should('not.be.visible');
  });
});
