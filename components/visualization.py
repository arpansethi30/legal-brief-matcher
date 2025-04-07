# File: components/visualization.py

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

class LegalNetworkVisualizer:
    """Advanced visualization for legal argument networks"""
    
    def __init__(self):
        # Define custom color scheme for legal documents
        self.colors = {
            'moving_brief': '#1E5AA8',  # Deep blue
            'response_brief': '#A83A1E',  # Deep red
            'edges': ['#E6F0FF', '#1E5AA8'],  # Light blue to deep blue gradient
            'edges_alt': ['#FFF0E6', '#A83A1E'],  # Light red to deep red (for hovering)
            'background': '#F5F5F5',
            'high_confidence': '#27AE60',  # Green for high confidence
            'medium_confidence': '#F39C12',  # Orange for medium confidence
            'low_confidence': '#C0392B'  # Red for low confidence
        }
        
        # Create custom colormap for edges
        self.edge_cmap = LinearSegmentedColormap.from_list('legal_edge', self.colors['edges'])
        self.edge_alt_cmap = LinearSegmentedColormap.from_list('legal_edge_alt', self.colors['edges_alt'])
    
    def create_network_visualization(self, matches, brief_pair):
        """Create network visualization of argument matches with enhanced styling"""
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes for moving brief arguments
        for i, arg in enumerate(brief_pair['moving_brief']['brief_arguments']):
            # Extract complexity if available
            complexity = arg.get('complexity', 50) if isinstance(arg, dict) else 50
            
            G.add_node(f"M{i}", 
                      label=arg['heading'],
                      type='moving',
                      index=i,
                      complexity=complexity)
        
        # Add nodes for response brief arguments
        for i, arg in enumerate(brief_pair['response_brief']['brief_arguments']):
            # Extract complexity if available
            complexity = arg.get('complexity', 50) if isinstance(arg, dict) else 50
            
            G.add_node(f"R{i}", 
                      label=arg['heading'],
                      type='response',
                      index=i,
                      complexity=complexity)
        
        # Add edges for matches
        for match in matches:
            # Extract match metrics
            confidence = match['confidence']
            shared_citations = len(match.get('shared_citations', []))
            
            G.add_edge(f"M{match['moving_index']}", 
                      f"R{match['response_index']}", 
                      weight=confidence,
                      citations=shared_citations,
                      explanation=match.get('explanation', ''))
        
        # Create positions
        pos = {}
        moving_nodes = [n for n in G.nodes() if n.startswith('M')]
        response_nodes = [n for n in G.nodes() if n.startswith('R')]
        
        # Position moving nodes on left with staggered arrangement
        for i, node in enumerate(moving_nodes):
            pos[node] = (-1, -i * 1.0)
        
        # Position response nodes on right with staggered arrangement
        for i, node in enumerate(response_nodes):
            pos[node] = (1, -i * 1.0)
        
        # Create figure with higher resolution
        fig, ax = plt.subplots(figsize=(12, 10), facecolor=self.colors['background'], dpi=100)
        
        # Configure for streamlit
        plt.tight_layout()
        plt.axis('off')
        
        # Node styling - size varies by complexity of argument
        node_sizes = [400 + (G.nodes[n].get('complexity', 50) * 2) for n in G.nodes()]
        node_colors = [self.colors['moving_brief'] if n.startswith('M') else 
                       self.colors['response_brief'] for n in G.nodes()]
        
        # Draw nodes with improved styling
        nodes = nx.draw_networkx_nodes(G, pos, 
                               node_color=node_colors,
                               node_size=node_sizes,
                               alpha=0.9,
                               edgecolors='white',
                               linewidths=1.5)
        
        # Add drop shadow effect to nodes - Apply to entire PathCollection instead of iterating
        nodes.set_path_effects([
            pe.withStroke(linewidth=5, foreground='darkgray', alpha=0.4)
        ])
        
        # Edge styling - width and color based on confidence and citation count
        for (u, v, data) in G.edges(data=True):
            # Calculate edge width based on confidence
            width = 1 + (data['weight'] * 4)
            
            # Calculate color based on confidence
            if data['weight'] >= 0.7:
                color = self.colors['high_confidence']
                alpha = 0.9
            elif data['weight'] >= 0.5:
                color = self.colors['medium_confidence']
                alpha = 0.8
            else:
                color = self.colors['low_confidence']
                alpha = 0.7
            
            # Create curved edges with custom styling
            edge = nx.draw_networkx_edges(
                G, pos, 
                edgelist=[(u, v)],
                width=width,
                edge_color=[color],
                alpha=alpha,
                connectionstyle="arc3,rad=0.2",
                arrowsize=15,
                arrowstyle='->'
            )
            
            # Add citation count indicator if there are shared citations
            if data.get('citations', 0) > 0:
                # Calculate midpoint of the edge
                mid_x = (pos[u][0] + pos[v][0]) / 2
                mid_y = (pos[u][1] + pos[v][1]) / 2
                
                # Adjust for curve
                mid_x += 0.1  # Offset for curve
                
                # Draw citation indicator
                citation_size = min(15, 8 + (data['citations'] * 2))
                citation_circle = plt.Circle((mid_x, mid_y), citation_size/250, 
                                            color='white', alpha=0.9, zorder=5)
                ax.add_patch(citation_circle)
                
                # Add citation count text
                plt.text(mid_x, mid_y, str(data['citations']), 
                        ha='center', va='center', 
                        fontsize=7, color='black', fontweight='bold',
                        zorder=6)
        
        # Draw labels with white text for contrast
        label_pos = {k: (v[0], v[1] - 0.05) for k, v in pos.items()}
        labels = {n: G.nodes[n]['label'] for n in G.nodes()}
        
        # Truncate long labels
        truncated_labels = {}
        for node, label in labels.items():
            if len(label) > 30:
                truncated_labels[node] = label[:27] + "..."
            else:
                truncated_labels[node] = label
        
        # Draw labels with better styling
        nx.draw_networkx_labels(
            G, label_pos, 
            labels=truncated_labels,
            font_size=9, 
            font_color='white',
            font_weight='bold',
            horizontalalignment='center',
            bbox=dict(boxstyle="round,pad=0.3", fc=(0, 0, 0, 0.3), ec="none", alpha=0.7)
        )
        
        # Add title with better styling
        plt.title("Legal Brief Argument Network", 
                 fontsize=16, 
                 fontweight='bold', 
                 color='#333333',
                 pad=20)
        
        # Add confidence scale legend
        ax = plt.gca()
        ax.text(0.01, 0.03, "Match Confidence:", transform=ax.transAxes, fontsize=10, fontweight='bold')
        
        # Add colored legend items
        legend_x = 0.15
        for label, color, conf in [
            ("High (≥0.7)", self.colors['high_confidence'], 0.8),
            ("Medium (≥0.5)", self.colors['medium_confidence'], 0.6),
            ("Low (<0.5)", self.colors['low_confidence'], 0.4)
        ]:
            ax.plot([legend_x, legend_x + 0.08], [0.03, 0.03], 
                   transform=ax.transAxes,
                   linewidth=2 + (conf * 3), 
                   color=color,
                   solid_capstyle='round')
            
            ax.text(legend_x + 0.1, 0.03, label, 
                   transform=ax.transAxes, 
                   fontsize=8, 
                   verticalalignment='center')
            
            legend_x += 0.25
        
        # Add brief type legend
        legend_y = 0.01
        for label, color, marker in [
            ("Moving Brief", self.colors['moving_brief'], 'o'),
            ("Response Brief", self.colors['response_brief'], 'o')
        ]:
            ax.plot(0.01, legend_y, 
                   transform=ax.transAxes,
                   marker=marker, 
                   markersize=8, 
                   color=color)
            
            ax.text(0.04, legend_y, label, 
                   transform=ax.transAxes, 
                   fontsize=8, 
                   verticalalignment='center')
            
            legend_y -= 0.02
        
        # Add citation indicator legend
        ax.add_patch(plt.Circle((0.77, 0.02), 0.008, 
                              color='white', 
                              transform=ax.transAxes,
                              zorder=5))
        plt.text(0.77, 0.02, "2", 
                ha='center', va='center', 
                fontsize=6, color='black',
                transform=ax.transAxes,
                zorder=6, fontweight='bold')
        ax.text(0.8, 0.02, "Shared Citations", 
               transform=ax.transAxes, 
               fontsize=8, 
               verticalalignment='center')
        
        return fig
    
    def create_interactive_table(self, matches):
        """Create interactive table of matches with enhanced metrics"""
        # Format data for table
        table_data = []
        for i, match in enumerate(matches):
            # Format confidence with color
            if match['confidence'] >= 0.7:
                confidence_color = self.colors['high_confidence']
            elif match['confidence'] >= 0.5:
                confidence_color = self.colors['medium_confidence']
            else:
                confidence_color = self.colors['low_confidence']
            
            confidence = f"<span style='color:{confidence_color};font-weight:bold'>{match['confidence']:.2f}</span>"
            
            # Format other metrics
            shared_citations = len(match.get('shared_citations', []))
            shared_terms = len(match.get('shared_terms', []))
            
            # Extract more metrics if available
            counter_strength = match.get('counter_strength', '')
            if counter_strength:
                strength_display = f"{counter_strength}/10"
            else:
                strength_display = "N/A"
            
            # Get argument type if available
            arg_type = match.get('argument_type', '').capitalize()
            
            # Create row
            row = {
                'Match #': i+1,
                'Moving Brief Heading': self._truncate_text(match['moving_heading'], 40),
                'Response Brief Heading': self._truncate_text(match['response_heading'], 40),
                'Confidence': confidence,
                'Type': arg_type,
                'Shared Citations': shared_citations,
                'Shared Terms': shared_terms,
                'Counter Strength': strength_display,
                'Explanation': self._truncate_text(match.get('explanation', 'Semantic similarity'), 60)
            }
            table_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Return for streamlit to render
        return df
    
    def highlight_matching_text(self, text, terms=None, citations=None):
        """Highlight terms and citations in text with improved styling"""
        highlighted = text
        
        # Highlight citations with improved styling
        if citations:
            for citation in citations:
                highlighted = highlighted.replace(
                    citation, 
                    f"<span style='background-color:#e8f4f8; padding:2px 4px; border-radius:3px; font-family:monospace; font-weight:bold; color:#2C3E50;'>{citation}</span>"
                )
        
        # Highlight legal terms with improved styling
        if terms:
            for term in terms:
                if term in highlighted.lower():
                    # Case-insensitive replacement
                    idx = highlighted.lower().find(term)
                    original_term = highlighted[idx:idx+len(term)]
                    highlighted = highlighted.replace(
                        original_term,
                        f"<span style='background-color:#f5f5dc; padding:2px 4px; border-radius:3px; font-style:italic; color:#6E4C1E; font-weight:bold;'>{original_term}</span>"
                    )
        
        return highlighted
    
    def create_confidence_breakdown(self, match):
        """Create visual breakdown of confidence factors"""
        if 'confidence_factors' not in match:
            return None
        
        factors = match['confidence_factors']
        
        # Create a simple bar chart of confidence factors
        labels = []
        values = []
        colors = []
        
        # Add base similarity
        labels.append('Base Similarity')
        values.append(factors.get('base_similarity', {}).get('value', 0) if isinstance(factors.get('base_similarity'), dict) else factors.get('base_similarity', 0))
        colors.append('#3498DB')  # Blue
        
        # Add positive factors
        for factor, color in [
            ('citation_boost', '#27AE60'),  # Green
            ('heading_match', '#9B59B6'),  # Purple
            ('legal_terminology', '#F1C40F'),  # Yellow
            ('pattern_match', '#E67E22'),  # Orange
            ('precedent_impact', '#16A085')  # Teal
        ]:
            if factor in factors:
                factor_value = factors.get(factor, {}).get('value', 0) if isinstance(factors.get(factor), dict) else factors.get(factor, 0)
                if factor_value > 0:
                    labels.append(factor.replace('_', ' ').title())
                    values.append(factor_value)
                    colors.append(color)
        
        # Add penalties
        for factor in ['length_penalty']:
            if factor in factors:
                factor_value = factors.get(factor, {}).get('value', 0) if isinstance(factors.get(factor), dict) else factors.get(factor, 0)
                if factor_value > 0:
                    labels.append(factor.replace('_', ' ').title())
                    values.append(-factor_value)  # Negative value for penalty
                    colors.append('#E74C3C')  # Red for penalties
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 2.5))
        
        # Create horizontal bars
        y_pos = range(len(labels))
        bars = ax.barh(y_pos, values, color=colors)
        
        # Add labels to bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width >= 0 else width - 0.05
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                   va='center', ha='left' if width >= 0 else 'right', color='black', fontsize=8)
        
        # Add final score line
        final_score = factors.get('final_score', {}).get('value', 0) if isinstance(factors.get('final_score'), dict) else factors.get('final_score', 0)
        if final_score > 0:
            ax.axvline(x=final_score, color='black', linestyle='--', linewidth=1)
            ax.text(final_score, len(labels), f"Final: {final_score:.2f}", 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Contribution to Confidence Score')
        ax.set_title('Match Confidence Factor Breakdown', fontsize=12)
        
        # Set x-axis limits with some padding
        ax.set_xlim(-0.1, min(1.1, max(values) * 1.2))
        
        plt.tight_layout()
        return fig
    
    def create_shared_citation_table(self, shared_citations):
        """Create a formatted table of shared citations"""
        if not shared_citations:
            return pd.DataFrame()
        
        # Create a dataframe with citation information
        citation_data = []
        
        for citation in shared_citations:
            # Determine citation type
            if 'v.' in citation:
                citation_type = 'Case Law'
            elif '§' in citation:
                citation_type = 'Statute'
            elif 'Const.' in citation:
                citation_type = 'Constitutional'
            else:
                citation_type = 'Other'
            
            citation_data.append({
                'Citation': citation,
                'Type': citation_type
            })
        
        return pd.DataFrame(citation_data)
    
    def _truncate_text(self, text, max_length=50):
        """Truncate text to specified length"""
        if len(text) > max_length:
            return text[:max_length-3] + "..."
        return text
    
    def create_opposing_counsel_insight(self, match):
        """Create a visualization of opposing counsel insights based on match analysis"""
        # Initialize figure
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f9f9f9')
        
        # Set up basic styling
        plt.title("Opposing Counsel Insight", fontsize=16, color='#333333', fontweight='bold')
        ax.set_facecolor('#f9f9f9')
        
        # Get counter analysis from the match if available
        counter_analysis = match.get('counter_analysis', {})
        precedent_analysis = match.get('precedent_analysis', {})
        
        # Set up the sections we'll display
        insight_sections = [
            "Argument Strengths & Weaknesses",
            "Precedent Analysis",
            "Potential Counterarguments"
        ]
        
        # Create a table structure
        table_data = []
        
        # 1. Argument Strengths & Weaknesses
        if counter_analysis and counter_analysis.get('strengths') and counter_analysis.get('weaknesses'):
            strengths = counter_analysis.get('strengths', [])
            weaknesses = counter_analysis.get('weaknesses', [])
            
            strengths_text = "\n".join([f"• {s}" for s in strengths if s])
            weaknesses_text = "\n".join([f"• {s}" for s in weaknesses if s])
            
            # Add strength/weakness rows
            table_data.append([
                "STRENGTHS",
                strengths_text
            ])
            
            table_data.append([
                "WEAKNESSES",
                weaknesses_text
            ])
        else:
            table_data.append([
                "ANALYSIS",
                "• The response argument addresses key points from the moving brief\n• Consider strengthening citation support\n• Watch for logical fallacies in reasoning"
            ])
        
        # 2. Precedent Analysis
        if precedent_analysis:
            relative_strength = precedent_analysis.get('relative_strength', 0)
            analysis_text = precedent_analysis.get('analysis', '')
            
            # Add precedent analysis with visual indicator
            strength_indicator = ""
            if relative_strength > 0.3:
                strength_indicator = "⚠️ WEAKER PRECEDENTS THAN RESPONSE"
            elif relative_strength < -0.3:
                strength_indicator = "✓ STRONGER PRECEDENTS THAN RESPONSE"
            else:
                strength_indicator = "➖ COMPARABLE PRECEDENT STRENGTH"
            
            table_data.append([
                "PRECEDENT STRENGTH",
                f"{strength_indicator}\n{analysis_text}"
            ])
            
            # Add key precedents
            common_precedents = precedent_analysis.get('common_key_precedents', [])
            if common_precedents:
                precedent_list = "\n".join([f"• {p['name']} ({p['year']})" for p in common_precedents[:3]])
                table_data.append([
                    "KEY SHARED PRECEDENTS",
                    precedent_list
                ])
        
        # 3. Strategic Recommendations
        # This section provides actual insights for opposing counsel
        counter_quality = counter_analysis.get('counter_quality_score', 0.5)
        reasoning = counter_analysis.get('reasoning', '')
        
        recommendations = []
        
        # Generate recommendations based on counter argument quality
        if counter_quality < 0.4:
            recommendations.append("The opposition's counter-argument is weak. Emphasize this point in reply.")
        elif counter_quality > 0.7:
            recommendations.append("Opposition has strong counter-arguments. Consider refining your position.")
        
        # Generate recommendations based on precedent strength
        if precedent_analysis:
            relative_strength = precedent_analysis.get('relative_strength', 0)
            if relative_strength > 0.3:
                recommendations.append("Opposition cites stronger precedents. Consider distinguishing these cases.")
            elif relative_strength < -0.3:
                recommendations.append("Your precedents are stronger. Emphasize their authority in reply.")
        
        # Add some general strategic recommendations
        if match.get('shared_citations'):
            num_shared = len(match.get('shared_citations', []))
            recommendations.append(f"Both briefs cite {num_shared} common authorities. Address opposition's interpretation.")
        
        # Add custom recommendations based on the combination of factors
        if counter_quality > 0.6 and precedent_analysis.get('relative_strength', 0) > 0.2:
            recommendations.append("⚠️ VULNERABLE POSITION: Opposition has both strong arguments and precedents.")
        elif counter_quality < 0.4 and precedent_analysis.get('relative_strength', 0) < -0.2:
            recommendations.append("✓ STRONG POSITION: Your argument has stronger reasoning and precedential support.")
        
        # Add recommendations to table
        recommendation_text = "\n".join([f"• {r}" for r in recommendations])
        table_data.append([
            "STRATEGIC RECOMMENDATIONS",
            recommendation_text
        ])
        
        # Create the table
        table = ax.table(
            cellText=table_data,
            colWidths=[0.25, 0.65],
            loc='center',
            cellLoc='left'
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style header cells differently
        for i, cell in enumerate(table._cells):
            if cell[1] == 0:  # Column 0 - headers
                table._cells[cell].set_text_props(
                    fontweight='bold', 
                    color='white'
                )
                table._cells[cell].set_facecolor('#1E5AA8')
                table._cells[cell].set_edgecolor('white')
            else:
                table._cells[cell].set_edgecolor('#dddddd')
                
            if i % 2 == 0:  # Add alternating row colors
                if cell[1] == 1:  # Only apply to content cells
                    table._cells[cell].set_facecolor('#f0f0f0')
        
        # Remove axes
        ax.axis('off')
        ax.axis('tight')
        
        return fig