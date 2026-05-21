import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import App from './App';

describe('App Component', () => {
  it('renders landing page by default', () => {
    render(<App />);
    // The landing page should be visible; look for a heading or button
    const main = screen.getByRole('main');
    expect(main).toBeTruthy();
  });

  it('renders a main container', () => {
    render(<App />);
    // If component is working, main element should exist
    const main = screen.getByRole('main');
    expect(main).toBeDefined();
    expect(main.className).toContain('min-h-screen');
  });
});
