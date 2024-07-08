describe('template spec', () => {
  it('passes', () => {
      cy.visit('_reports/employee_salaries.html');
      cy.get('skrub-table-report').shadow().find('td').first().click();
  })
})
